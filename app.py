import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from bayes_models import calc_gamma_params, gamma_poisson_posterior

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Inferencia Bayesiana - Turismo",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS PERSONALIZADO (Wow Factor Estético) ---
st.markdown("""
<style>
    /* Estilos Premium para la App */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E2129;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #A0AEC0;
    }
    h1, h2, h3 {
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECCTION ---
st.title("🍲 Análisis Bayesiano en Turismo")
st.markdown("**Villa de Leyva, Colombia: conteos y decisiones**")
st.markdown("---")

# --- SECCIÓN ÚNICA ---
st.markdown("### 📊 Parte A: Tamales (Gamma-Poisson)")
st.header("Decisión de Porciones: ¿Cuántos tamales comprar?")
st.markdown("El número de turistas $Y$ sigue una distribución Poisson($\lambda$). Usamos Inferencia Bayesiana para actualizar nuestra creencia sobre $\lambda$ (tasa promedio de turistas diaria).")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Configurar Creencia Previa (Prior)")
    prior_type_a = st.radio("Elige la fuente de la Prior:", ["Priori Informada", "Prior Débil", "Libre"])
    
    if prior_type_a == "Priori Informada":
        alpha_prior = 12.5
        beta_prior = 0.25
        st.info(f"Prior fuerte: Gamma(α={alpha_prior:.1f}, β={beta_prior:.2f})")
    elif prior_type_a == "Prior Débil":
        alpha_prior = 16.0
        beta_prior = 0.2
        st.warning(f"Prior débil: Gamma(α={alpha_prior:.1f}, β={beta_prior:.1f})")
    else:
        mu_prior_a = st.number_input("Media Esperada ($\mu$)", value=40, step=5)
        var_prior_a = st.number_input("Varianza ($\sigma^2$)", value=100, step=10)
        alpha_prior, beta_prior = calc_gamma_params(mu_prior_a, var_prior_a)
        st.success(f"Prior personalizada (calculada): Gamma(α={alpha_prior:.2f}, β={beta_prior:.2f})")
    
    st.subheader("2. Ingresar Nuevos Datos (Likelihood)")
    y_obs_a = st.number_input("Turistas contados hoy ($y$)", value=55, step=1)
    
    # Cálculos Matemáticos:
    alpha_post, beta_post = gamma_poisson_posterior(alpha_prior, beta_prior, y_obs_a)
    
    # Decisiones:
    media_post = alpha_post / beta_post
    # Predictiva Posterior (Poisson-Gamma es Binomial Negativa)
    p_nb = beta_post / (1 + beta_post)
    r_nb = alpha_post
    pred_95 = stats.nbinom.ppf(0.95, r_nb, p_nb)
    
    st.markdown("---")
    st.subheader("Recomendación Final")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Decisión "Eficiente" (Promedio esperado):</div>
        <div class="metric-value">{int(round(media_post))} Tamales</div>
        <br/>
        <div class="metric-label">Decisión "Prudente" (Cubre el 95% de escenarios):</div>
        <div class="metric-value">{int(pred_95)} Tamales</div>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.subheader("Visualización del Aprendizaje Bayesiano")
    
    # Encontrar la media para establecer un rango de X lógico
    media_graf = alpha_prior / beta_prior
    x_lambda = np.linspace(0, max(media_graf, y_obs_a) + 50, 500)
    
    # PDFs
    pdf_prior = stats.gamma.pdf(x_lambda, a=alpha_prior, scale=1/beta_prior)
    pdf_post = stats.gamma.pdf(x_lambda, a=alpha_post, scale=1/beta_post)
    
    # Likelihood escalada (Poisson) - Escalada a máximo 1
    x_discrete = np.arange(0, max(media_graf, y_obs_a) + 50)
    likelihood = stats.poisson.pmf(y_obs_a, x_discrete)
    likelihood_scaled = likelihood / np.max(likelihood)
    
    fig = go.Figure()
    
    # Prior
    fig.add_trace(go.Scatter(x=x_lambda, y=pdf_prior, mode='lines', name='Prior (Creencia Antes)', line=dict(color='#FFA726', width=3), fill='tozeroy', fillcolor='rgba(255, 167, 38, 0.2)'))
    
    # Posterior
    fig.add_trace(go.Scatter(x=x_lambda, y=pdf_post, mode='lines', name='Posterior (Creencia Actualizada)', line=dict(color='#4CAF50', width=3), fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.3)'))
    
    # Likelihood
    fig.add_trace(go.Scatter(x=x_discrete, y=likelihood_scaled * max(max(pdf_prior), max(pdf_post)), mode='lines', name='Likelihood (Lo que dice el Dato)', line=dict(color='#29B6F6', width=2, dash='dash')))
    
    fig.update_layout(
        title="Gráfica A: ¿Quién manda más, la Prior o el Dato?",
        xaxis_title="Tasa de turistas (λ)",
        yaxis_title="Densidad de Probabilidad",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- GRÁFICA PREDICTIVA ---
    st.subheader("Predicción Asistida: Distribución de Turistas Mañana")
    
    # Rango para Y_mañana
    y_futuro = np.arange(0, int(pred_95) + max(20, int(pred_95 * 0.3)))
    pmf_pred = stats.nbinom.pmf(y_futuro, r_nb, p_nb)
    
    fig_pred = go.Figure()
    
    # Barras normales
    fig_pred.add_trace(go.Bar(
        x=y_futuro, 
        y=pmf_pred, 
        name='Probabilidad',
        marker_color='rgba(158, 158, 158, 0.6)'
    ))
    
    # Resaltado del intervalo 95% (desde 0 hasta pred_95)
    y_interval = y_futuro[y_futuro <= pred_95]
    pmf_interval = pmf_pred[y_futuro <= pred_95]
    
    fig_pred.add_trace(go.Bar(
        x=y_interval, 
        y=pmf_interval, 
        name='Cubre el 95% de probabilidad',
        marker_color='rgba(233, 30, 99, 0.8)'
    ))
    
    # Linea de decisión "Eficiente" (Media)
    fig_pred.add_vline(x=media_post, line_width=3, line_dash="dash", line_color="#4CAF50", annotation_text="Eficiente (Media)")
    # Linea de decisión "Prudente"
    fig_pred.add_vline(x=pred_95, line_width=3, line_dash="solid", line_color="#E91E63", annotation_text="Prudente (95%)")
    
    fig_pred.update_layout(
        title="Predictiva Posterior (Binomial Negativa)",
        xaxis_title="Número de Turistas Mañana",
        yaxis_title="Probabilidad",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        showlegend=False
    )
    st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 14px;'>Dashboard desarrollado para proyecto final de Inferencia Bayesiana.</p>", unsafe_allow_html=True)
