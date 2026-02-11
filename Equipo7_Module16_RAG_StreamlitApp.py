import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile # Para manejar archivos subidos
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================================
# para poder subir los archivos
# ============================================================================


def process_uploaded_file(uploaded_file):
    """Procesa el PDF y genera fragmentos para la base vectorial (RAG Core)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        # 1. Cargar y dividir
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        # 2. Crear Vector Store
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # 3. Guardar en session_state
        st.session_state.vector_db = vector_db
        if uploaded_file.name not in st.session_state.processed_files:
            st.session_state.processed_files.append(uploaded_file.name)
        st.session_state.rag_metrics['total_chunks'] += len(chunks)
        st.session_state.rag_metrics['last_upload'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return True, len(chunks)
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================================================
# LLM CONFIGURATION & CACHING
# ============================================================================

@st.cache_resource
def load_llm():
    """Carga el cliente de OpenAI usando la clave del sistema."""
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

@st.cache_data
def compute_explanation(_input_text: str, _response: str) -> dict:
    """
    Calcula metricas de explicabilidad.
    Se define aqui para evitar NameError en la funcion siguiente.
    """
    return {
        'input_tokens': len(_input_text.split()),
        'response_tokens': len(_response.split()),
        'confidence': 0.95,
        'top_features': ['Analisis de texto', 'Contexto', 'Prompt'],
        'explanation': 'Explicacion base cargada correctamente.'
    }

def generate_response(message: str, temperature: float = 0.7) -> tuple:
    """
    Genera respuesta real usando OpenAI y mide el tiempo.
    """
    start_time = time.time()
    
    # 1. Cargamos el cliente real
    client = load_llm()
    
    # --- BLOQUE RAG (INDISPENSABLE PARA M16) *******************************************************************************
    if st.session_state.vector_db is not None:
        # Busca los 3 fragmentos más parecidos a la pregunta
        docs = st.session_state.vector_db.similarity_search(message, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        system_content = f"Usa este contexto para responder:\n{context}\n\nInstruccion: {st.session_state.preferences['system_prompt']}"
    else:
        system_content = st.session_state.preferences['system_prompt']
    # ****************************************************************************************************************
     
    
    # 2. Preparamos los mensajes para OpenAI
    messages = [{"role": "system", "content": st.session_state.preferences['system_prompt']}]
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    
    # 3. Llamada a la API de OpenAI
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=st.session_state.preferences['max_tokens']
        )
        response = completion.choices[0].message.content
    except Exception as e:
        response = f"Error en la API: {str(e)}"
    
    # 4. Calculamos la explicacion y tiempos
    explanation = compute_explanation(message, response)
    response_time = time.time() - start_time
    
    # Actualizar metricas
    st.session_state.metrics['total_messages'] += 1
    st.session_state.metrics['avg_response_time'] = (
        (st.session_state.metrics['avg_response_time'] * (st.session_state.metrics['total_messages'] - 1) + response_time) /
        st.session_state.metrics['total_messages']
    )
    
    return response, explanation, response_time

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Feedback database (in-memory for demo)
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = []
    
    # User preferences
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'temperature': 0.7,
            'max_tokens': 500,
            'system_prompt': 'You are a helpful AI assistant.'
        }
    
    # Current explanation
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    
    # Performance metrics
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'total_messages': 0,
            'avg_response_time': 0,
            'total_feedback': 0
        }
#******************************************************************************************
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'rag_metrics' not in st.session_state:
        st.session_state.rag_metrics = {'total_chunks': 0, 'last_upload': None}
#**********************************************************************************************


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_response(message: str, temperature: float = 0.7) -> tuple:
    """
    Genera respuesta real usando OpenAI y mide el tiempo.
    """
    start_time = time.time()
    
    # 1. Cargamos el cliente real configurado anteriormente
    client = load_llm()
    
    # 2. Preparamos el historial para OpenAI
    messages = [{"role": "system", "content": st.session_state.preferences['system_prompt']}]
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    
    # 3. Llamada correcta a la API
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=st.session_state.preferences['max_tokens']
        )
        response = completion.choices[0].message.content
    except Exception as e:
        response = f"Error en la API: {str(e)}"
    
    # 4. Calculamos la explicacion y tiempos
    explanation = compute_explanation(message, response)
    response_time = time.time() - start_time
    
    # Actualizar metricas
    st.session_state.metrics['total_messages'] += 1
    st.session_state.metrics['avg_response_time'] = (
        (st.session_state.metrics['avg_response_time'] * (st.session_state.metrics['total_messages'] - 1) + response_time) /
        st.session_state.metrics['total_messages']
    )
    
    return response, explanation, response_time

def save_feedback(message: str, response: str, rating: str, comment: str):
    """Guarda el feedback del usuario en el session state."""
    feedback_entry = {
        'timestamp': datetime.now(),
        'message': message,
        'response': response,
        'rating': rating,
        'comment': comment
    }
    
    st.session_state.feedback_db.append(feedback_entry)
    st.session_state.metrics['total_feedback'] += 1
    
    # Nota: Asegurate de definir load_feedback_data mas adelante 
    # o comenta la siguiente linea si marca error.
    # load_feedback_data.clear()
    
# ============================================================================
# PAGE: CHAT INTERFACE
# ============================================================================

def page_chat():
    """Main chat interface page."""
    st.title("🤖 AI Chat with Explainability")
    st.markdown("Ask questions and receive transparent, explainable responses.")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.preferences['temperature'],
            step=0.1,
            help="Higher values make output more random"
        )
        st.session_state.preferences['temperature'] = temperature
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=st.session_state.preferences['max_tokens'],
            step=50
        )
        st.session_state.preferences['max_tokens'] = max_tokens
        
        with st.expander("System Prompt"):
            system_prompt = st.text_area(
                "System Instructions",
                value=st.session_state.preferences['system_prompt'],
                height=100
            )
            st.session_state.preferences['system_prompt'] = system_prompt
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_explanation = None
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation")
        
        # Display chat history
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Generate response
            with st.spinner("Thinking..."):
                response, explanation, response_time = generate_response(
                    prompt,
                    temperature
                )
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.current_explanation = {
                'input': prompt,
                'output': response,
                'details': explanation,
                'response_time': response_time
            }
            
            # Display assistant message
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            st.rerun()
    
    with col2:
        st.subheader("Explainability")
        
        if st.session_state.current_explanation:
            exp = st.session_state.current_explanation
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{exp['details']['confidence']:.2f}")
            with col_b:
                st.metric("Response Time", f"{exp['response_time']:.2f}s")
            
            st.divider()
            
            # Explanation details
            st.markdown("**Key Factors:**")
            for feature in exp['details']['top_features']:
                st.markdown(f"- {feature}")
            
            st.divider()
            
            # Feedback section
            st.markdown("**Provide Feedback:**")
            
            rating = st.radio(
                "Rate this response:",
                options=["👍 Helpful", "👎 Not Helpful"],
                key=f"rating_{len(st.session_state.messages)}"
            )
            
            comment = st.text_area(
                "Optional comment:",
                placeholder="What was good or could be improved?",
                key=f"comment_{len(st.session_state.messages)}"
            )
            
            if st.button("Submit Feedback", use_container_width=True):
                save_feedback(
                    exp['input'],
                    exp['output'],
                    rating,
                    comment
                )
                st.success("✅ Feedback saved!")
        else:
            st.info("Send a message to see explanation and provide feedback.")

# ============================================================================
# PAGE: EXPLAINABILITY ANALYSIS
# ============================================================================

def page_explainability():
    """Pagina de analisis detallado de explicabilidad."""
    st.title("Explainability Analysis")
    st.markdown("Analisis profundo de las decisiones del modelo y patrones de comportamiento.")
    
    if not st.session_state.messages:
        st.info("No hay conversaciones aun. Ve a la pagina de Chat para comenzar.")
        return
    
    # Obtener conversaciones recientes
    conversations = []
    for i in range(0, len(st.session_state.messages), 2):
        if i+1 < len(st.session_state.messages):
            conversations.append({
                'user': st.session_state.messages[i]['content'],
                'assistant': st.session_state.messages[i+1]['content']
            })
    
    if not conversations:
        st.warning("No se encontraron conversaciones completas.")
        return
    
    # Seleccionar conversacion para analizar
    selected_idx = st.selectbox(
        "Seleccione la conversacion a analizar:",
        range(len(conversations)),
        format_func=lambda i: f"Conv {i+1}: {conversations[i]['user'][:50]}..."
    )
    
    selected_conv = conversations[selected_idx]
    
    # Mostrar conversacion
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entrada del Usuario")
        st.markdown(f"```\n{selected_conv['user']}\n```")
        
        st.subheader("Respuesta del Modelo")
        st.markdown(f"```\n{selected_conv['assistant']}\n```")
    
    with col2:
        st.subheader("Explicabilidad")
        
        # Calcular explicacion
        exp = compute_explanation(selected_conv['user'], selected_conv['assistant'])
        
        # Mostrar metricas
        st.metric("Tokens de Entrada", exp['input_tokens'])
        st.metric("Tokens de Respuesta", exp['response_tokens'])
        st.metric("Confianza", f"{exp['confidence']:.2%}")
        
        st.divider()
        
        st.markdown("**Factores Clave:**")
        for i, feature in enumerate(exp['top_features'], 1):
            st.markdown(f"{i}. {feature}")
    
    # Seccion de visualizacion
    st.divider()
    st.subheader("Visualizacion de Importancia de Caracteristicas")
    
    # Datos de importancia simulados
    features = ['Longitud de entrada', 'Complejidad', 'Relevancia de contexto', 'Coincidencia semantica', 'Reconocimiento de patrones']
    importance = [0.25, 0.20, 0.30, 0.15, 0.10]
    
    fig = go.Figure(data=[
        go.Bar(x=features, y=importance, marker_color='lightblue')
    ])
    fig.update_layout(
        title="Importancia de caracteristicas para la respuesta seleccionada",
        xaxis_title="Caracteristicas",
        yaxis_title="Puntaje de Importancia",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tarea del equipo: Implementar visualizaciones reales de SHAP o LIME aqui
    st.info("Tarea del equipo: Implementar visualizaciones reales de SHAP o LIME aqui.")

# ============================================================================
# PAGE: FEEDBACK DASHBOARD
# ============================================================================

def page_feedback():
    """Pagina de monitoreo de calidad y feedback de usuarios."""
    st.title("Feedback Dashboard")
    st.markdown("Monitoreo de comentarios de usuarios y calidad de respuestas.")
    
    # Carga de datos de feedback
    feedback_df = load_feedback_data()
    
    if feedback_df.empty:
        st.info("No se ha recopilado feedback aun. Interactue con el chat para generar datos.")
        return
    
    # Metricas de resumen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Feedback Total", len(feedback_df))
    
    with col2:
        # Busqueda flexible por palabra clave para evitar dependencia de iconos
        positive = len(feedback_df[feedback_df['rating'].str.contains('Helpful|Positivo', case=False)])
        st.metric("Positivo", positive)
    
    with col3:
        negative = len(feedback_df[feedback_df['rating'].str.contains('Not Helpful|Negativo', case=False)])
        st.metric("Negativo", negative)
    
    with col4:
        if len(feedback_df) > 0:
            satisfaction = (positive / len(feedback_df)) * 100
            st.metric("Satisfaccion", f"{satisfaction:.1f}%")
    
    st.divider()
    
    # Visualizacion de feedback
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribucion de Feedback")
        
        rating_counts = feedback_df['rating'].value_counts()
        fig = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Feedback Positivo vs Negativo"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feedback a lo largo del tiempo")
        
        feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
        daily_counts = feedback_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Cantidad de feedback por dia",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Feedback reciente
    st.subheader("Feedback Reciente")
    
    # Preparacion de la tabla
    display_df = feedback_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['message'] = display_df['message'].str[:50] + '...'
    display_df['response'] = display_df['response'].str[:50] + '...'
    
    st.dataframe(
        display_df[['timestamp', 'message', 'response', 'rating', 'comment']],
        use_container_width=True,
        hide_index=True
    )
    
    # Exportacion de datos
    st.divider()
    if st.button("Exportar datos de Feedback"):
        csv = feedback_df.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"feedback_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
# ============================================================================
# PAGE: MONITORING
# ============================================================================

def page_monitoring():
    """Pagina de monitoreo del sistema y metricas de rendimiento."""
    st.title("System Monitoring")
    st.markdown("Seguimiento del rendimiento de la aplicacion y metricas de uso.")
    
    # Resumen de metricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mensajes Totales",
            st.session_state.metrics['total_messages']
        )
    
    with col2:
        st.metric(
            "Tiempo Promedio de Respuesta",
            f"{st.session_state.metrics['avg_response_time']:.2f}s"
        )
    
    with col3:
        st.metric(
            "Feedback Total",
            st.session_state.metrics['total_feedback']
        )
    
    st.divider()
    
    # Estado de la cache
    st.subheader("Estado de la Cache")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cache del Modelo**")
        st.success("Modelo cargado y almacenado en cache")
        st.markdown("**Cache de Datos de Feedback**")
        st.info("TTL: 1 hora")
    
    with col2:
        if st.button("Limpiar todas las caches", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches limpiadas correctamente")
            st.rerun()
    
    st.divider()
    
    # Inspeccion del estado de la sesion
    with st.expander("Inspeccionar Session State (Debug)"):
        st.json({
            'messages_count': len(st.session_state.messages),
            'feedback_count': len(st.session_state.feedback_db),
            'preferences': st.session_state.preferences,
            'metrics': st.session_state.metrics
        })
    
    st.divider()
    
    # Recomendaciones del sistema
    st.subheader("Recomendaciones de Optimizacion")
    
    if st.session_state.metrics['avg_response_time'] > 2.0:
        st.warning("El tiempo promedio de respuesta es alto. Considere implementar respuestas por streaming.")
    else:
        st.success("Los tiempos de respuesta estan dentro del rango aceptable.")
    
    if st.session_state.metrics['total_feedback'] < st.session_state.metrics['total_messages'] * 0.3:
        st.info("La tasa de feedback es baja. Considere hacer el formulario de feedback mas prominente.")
    else:
        st.success("Buena tasa de participacion de los usuarios.")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_feedback_data():
    """
    Carga los datos de feedback directamente sin cache para permitir
    actualizaciones en tiempo real.
    """
    if 'feedback_db' not in st.session_state or not st.session_state.feedback_db:
        return pd.DataFrame(columns=['timestamp', 'message', 'response', 'rating', 'comment'])
    
    return pd.DataFrame(st.session_state.feedback_db)


# ============================================================================
# PAGE: page_knowledge_base
# ============================================================================

#***************************************************************************************
def page_knowledge_base():
    """Nueva seccion para gestionar el conocimiento del RAG."""
    st.title("Knowledge Base Management")
    st.markdown("Cargue documentos PDF para alimentar el conocimiento del asistente.")
    uploaded_file = st.file_uploader("Seleccione un archivo PDF", type=["pdf"])
    if uploaded_file:
        if st.button("Procesar Documento", use_container_width=True):
            with st.spinner("Procesando fragmentos..."):
                success, result = process_uploaded_file(uploaded_file)
                if success: st.success(f"Exito: {result} fragmentos generados.")
                else: st.error(f"Error: {result}")
    
    st.divider()
    if st.session_state.processed_files:
        st.write("Archivos procesados:")
        for f in st.session_state.processed_files: st.text(f"- {f}")
        evidence = {"modulo": 16, "archivos": st.session_state.processed_files, "chunks": st.session_state.rag_metrics['total_chunks']}
        st.download_button("Exportar Evidencia M16 (JSON)", pd.Series(evidence).to_json(), "evidencia_m16.json")


#****************************************************************************************

# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================


def page_documentation():
    """Pagina de informacion del equipo y documentacion."""
    st.title("Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Acerca de", "Tecnico", "Equipo"])
    
    with tab1:
        st.markdown("""
        ## Acerca de esta Aplicacion
        
        El Trustworthy AI Explainer Dashboard es una interfaz integral para interactuar
        con aplicaciones de LLM manteniendo la transparencia y la responsabilidad.
        
        ### Caracteristicas
        
        - **Chat Interactivo**: IA conversacional con memoria de sesion.
        - **Explicabilidad**: Analisis en tiempo real de las decisiones del modelo.
        - **Sistema de Feedback**: Recopilacion de calificaciones y comentarios.
        - **Monitoreo**: Metricas de rendimiento y analisis de uso.
        - **Dashboard Multipagina**: Interfaz organizada para diferentes tareas.
        
        ### Como usar
        
        1. **Chat**: Ve a la pagina de Chat para interactuar con la IA.
        2. **Revision**: Revisa las explicaciones de cada respuesta.
        3. **Feedback**: Califica las respuestas para ayudar a mejorar el sistema.
        4. **Analisis**: Usa la pagina de Explainability para un analisis profundo.
        5. **Monitoreo**: Sigue el rendimiento del sistema en la pagina de Monitoring.
        """)
    
    with tab2:
        st.markdown("""
        ## Documentacion Tecnica
        
        ### Arquitectura
        
        ```
        Frontend: Streamlit Multi-page App
        Backend: LLM (OpenAI/Local/Hugging Face)
        Explainability: SHAP/LIME
        State: st.session_state
        Caching: @st.cache_resource, @st.cache_data
        ```
        
        ### Gestion de Estado
        
        Esta aplicacion utiliza `st.session_state` para persistir:
        - Historial de conversacion.
        - Preferencias del usuario.
        - Datos de feedback.
        - Metricas de rendimiento.
        
        ### Estrategia de Cache
        
        - `@st.cache_resource`: Carga del modelo (costoso, no serializado).
        - `@st.cache_data`: Carga de datos (serializable, con TTL).
        
        ### Despliegue
        
        ```bash
        # Local
        streamlit run streamlit_app_template.py
        
        # Docker
        docker build -t trustworthy-ai-dashboard .
        docker run -p 8501:8501 trustworthy-ai-dashboard
        ```
        """)
    
    with tab3:
        with tab2:
        st.markdown("""
        
       ## Informacion del Equipo
        
        ###Nombre del Equipo**: Equipo 7
        
        ##Integrantes**:
        - **ISIDRO NOEL GUERRERO** - Rol: Streamlit Architect
        - **JULIA DIAZ ESCOBAR** - Rol: Backend Integrator
        - **PEDRO MAYORGA ORTIZ** - Rol: Gradio Developer
        - **JULIO A. VALDEZ** - Rol: Explainability Specialist
        - **MISAEL CORRALES** - Rol: Deployment Specialist
        
        ##Modulo**: 16 - Retrieval Augmented Generation (RAG)
        
        ##Proyecto**: Trustworthy AI Explainer Dashboard
        
        ### Contribuciones
        
        - **Prototipo Gradio**: PEDRO MAYORGA ORTIZ (Pruebas funcionales iniciales)
        - **Dashboard Streamlit**: ISIDRO NOEL GUERRERO (Estructura y navegación)
        - **Integracion de Explicabilidad**: JULIO A. VALDEZ (Lógica de métricas y visualización)
        - **Despliegue y Backend**: MISAEL CORRALES y JULIA DIAZ ESCOBAR (Gestión de API y Hosting)
        """)
# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 Trustworthy AI")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📁 Knowledge Base", "🔍 Explainability", "📊 Feedback", "📈 Monitoring","📚 Documentation"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption(f"Session: {len(st.session_state.messages)} messages")
    
    # Route to selected page
    if page == "💬 Chat":
        page_chat()
    elif page == "📁 Knowledge Base":
        page_knowledge_base()
    elif page == "🔍 Explainability":
        page_explainability()
    elif page == "📊 Feedback":
        page_feedback()
    elif page == "📈 Monitoring":
        page_monitoring()
    elif page == "📚 Documentation":
        page_documentation()

if __name__ == "__main__":
    main()



