import streamlit as st
import fitz
import io, os, base64
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from mistralai.client import Mistral

st.set_page_config(page_title='InsightLens AI', page_icon='📄', layout='wide')
load_dotenv()

st.markdown('''
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid rgba(128,128,128,.22); border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2.25rem;}
.hero p {margin:.55rem 0 0 0; opacity:.72;}
.small-card {border:1px solid rgba(128,128,128,.2); border-radius:14px; padding:1rem;}
</style>
<div class="hero"><h1>InsightLens AI</h1><p>Ask questions across text and images inside your PDF using CLIP, FAISS, Mistral AI.</p></div>
''', unsafe_allow_html=True)

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model.eval()
    return model, processor

clip_model, clip_processor = load_clip()

def _clip_feature_tensor(output):
    """Return a tensor across different Transformers CLIP output formats."""
    if isinstance(output, torch.Tensor):
        return output

    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value

    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden[:, 0, :]

    raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")


def embed_image(image):
    inputs = clip_processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)
        features = _clip_feature_tensor(output)

    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.squeeze(0).cpu().numpy().astype("float32")


def embed_text(text):
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.no_grad():
        output = clip_model.get_text_features(**inputs)
        features = _clip_feature_tensor(output)

    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.squeeze(0).cpu().numpy().astype("float32")


def process_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    docs, embeddings, image_store = [], [], {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for page_no, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            chunks = splitter.split_documents([Document(page_content=text, metadata={'page': page_no + 1, 'type':'text'})])
            for chunk in chunks:
                docs.append(chunk); embeddings.append(embed_text(chunk.page_content))
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                raw = doc.extract_image(img[0])['image']
                pil = Image.open(io.BytesIO(raw)).convert('RGB')
                image_id = f'page_{page_no+1}_img_{img_index}'
                buff = io.BytesIO(); pil.save(buff, format='PNG')
                image_store[image_id] = base64.b64encode(buff.getvalue()).decode()
                docs.append(Document(page_content=f'[Image: {image_id}]', metadata={'page':page_no+1,'type':'image','image_id':image_id}))
                embeddings.append(embed_image(pil))
            except Exception:
                pass
    doc.close()
    if not docs:
        raise ValueError('No readable text or images were found in this PDF.')
    arr = np.array(embeddings)
    store = FAISS.from_embeddings(
        text_embeddings=[(d.page_content, e) for d,e in zip(docs, arr)],
        embedding=None,
        metadatas=[d.metadata for d in docs]
    )
    return store, image_store, docs

def multimodal_message(query, retrieved, image_store):
    content = [{"type": "text", "text": f"Question: {query}\n\nUse only the retrieved PDF context below. If the answer is not supported, say so."}]
    text_docs = [d for d in retrieved if d.metadata.get("type") == "text"]
    if text_docs:
        context = "\n\n".join(f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs)
        content.append({"type": "text", "text": f"\nText excerpts:\n{context}"})
    for d in retrieved:
        if d.metadata.get("type") == "image":
            image_id = d.metadata.get("image_id")
            if image_id in image_store:
                content.append({"type": "text", "text": f"Image from page {d.metadata['page']}:"})
                content.append({"type": "image_url", "image_url": f"data:image/png;base64,{image_store[image_id]}"})
    return [{"role": "user", "content": content}]


with st.sidebar:
    st.header('Document')
    uploaded = st.file_uploader('Upload a PDF', type=['pdf'])
    k = st.slider('Retrieved chunks', 2, 8, 5)
    st.caption('Tip: keep your MISTRAL_API_KEY in a .env file.')
    if st.button('Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if uploaded:
    signature = (uploaded.name, uploaded.size)
    if st.session_state.get('pdf_signature') != signature:
        with st.spinner('Reading text, extracting images, and building the vector index...'):
            try:
                vs, images, docs = process_pdf(uploaded.getvalue())
                st.session_state.vector_store = vs
                st.session_state.image_store = images
                st.session_state.pdf_signature = signature
                st.session_state.doc_count = len(docs)
                st.session_state.image_count = len(images)
                st.session_state.messages = []
            except Exception as e:
                st.error(f'Could not process PDF: {e}')
                st.stop()

    c1,c2,c3 = st.columns(3)
    c1.metric('File', uploaded.name)
    c2.metric('Indexed items', st.session_state.doc_count)
    c3.metric('Extracted images', st.session_state.image_count)

    st.subheader('Ask your document')
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    query = st.chat_input('e.g. What does the chart show about revenue trends?')
    if query:
        st.session_state.messages.append({'role':'user','content':query})
        with st.chat_message('user'):
            st.markdown(query)
        with st.chat_message('assistant'):
            if not os.getenv('MISTRAL_API_KEY'):
                st.error('MISTRAL_API_KEY is missing. Add it to your .env file and restart Streamlit.')
            else:
                with st.spinner('Retrieving text and visual context...'):
                    retrieved = st.session_state.vector_store.similarity_search_by_vector(embed_text(query), k=k)
                    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
                    response = client.chat.complete(
                        model="mistral-small-latest",
                        messages=multimodal_message(query, retrieved, st.session_state.image_store),
                    )
                    answer = response.choices[0].message.content
                    if not isinstance(answer, str):
                        answer = str(answer)
                st.markdown(answer)
                with st.expander('Retrieved evidence'):
                    for i,d in enumerate(retrieved,1):
                        kind=d.metadata.get('type','unknown').title(); page=d.metadata.get('page','?')
                        st.markdown(f'**{i}. {kind} · Page {page}**')
                        if kind == 'Text': st.caption(d.page_content[:500])
                        else:
                            image_id=d.metadata.get('image_id')
                            if image_id in st.session_state.image_store:
                                st.image(base64.b64decode(st.session_state.image_store[image_id]), width=320)
                st.session_state.messages.append({'role':'assistant','content':answer})
else:
    st.info('Upload a PDF from the sidebar to start.')
    st.markdown('### What this UI demonstrates')
    st.markdown('- Multimodal PDF ingestion (text + images)\n- CLIP embeddings for cross-modal retrieval\n- FAISS similarity search\n- Mistral-powered answers grounded in retrieved PDF context\n- Evidence display for better explainability')
