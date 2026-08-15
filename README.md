# MMM × MTA — Framework de Digital Analytics

Aplicativo Streamlit multi-página que materializa, com dados reais e Machine Learning, o framework
**MMM × MTA × Testes A/B**: a camada estratégica (Marketing Mix Modeling), a camada tática
(Multi-Touch Attribution + propensão preditiva), a camada de validação causal (Testes A/B e
geo-holdout) e o loop de governança que fecha o ciclo (Learning Repository).

---

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11+. As dependências opcionais (`pymc`, `arviz`, `xgboost`, `shap`) estão comentadas no
`requirements.txt` — o app detecta a ausência delas em tempo de execução e degrada graciosamente
(modo frequentista no MMM, `GradientBoostingClassifier` no lugar do XGBoost, importância nativa no
lugar do SHAP).

---

## Estrutura

```
app.py                     entrypoint (st.Page + st.navigation), tema e sidebar global
.streamlit/config.toml     tema visual (navy/teal/azul/dourado)
data/                      os 3 datasets
src/
  data_loader.py           leitura e limpeza cacheadas (st.cache_data)
  mmm/
    transforms.py          adstock geométrico, saturação Hill, formas funcionais
    model.py               fit regularizado, decomposição, curvas de resposta, VIF, Bayesiano opcional
    optimizer.py           otimização de budget (SLSQP) e cenário "E se"
  mta/
    journey_sim.py         simulação determinística de jornadas multi-touch
    heuristics.py          first/last-click, linear, time-decay, position-based
    markov.py              cadeia de Markov + removal effect
    shapley.py             Shapley Value exato e Monte Carlo
    propensity_model.py    pipeline de classificação (LogReg/RF/XGB) + SHAP
  abtest/
    power.py               duração e tamanho de amostra
    frequentist.py         z-test de proporções, qui-quadrado, ICs
    bayesian.py            Beta-Binomial, P(B>A)
    sequential.py          SPRT simplificado
    geo_holdout.py         holdout simulado + diferença-em-diferenças
  viz/charts.py            wrappers Plotly com o tema do app
  utils/                   styling.py (paleta/CSS) e repository.py (Learning Repository)
pages/                     as 11 páginas
```

---

## Página → conceito do framework

| Página | Camada | O que faz | Dataset |
|---|---|---|---|
| 🏠 Home | Governança | Resumo executivo, KPIs dos 3 datasets, diagrama do loop | todos |
| 📊 MMM Explorer | Estratégico | Data Quality (missings, outliers), análise uni e bivariada | `mmm_dataset.csv` |
| 🧪 MMM Modelagem | Estratégico | Adstock, saturação, regressão regularizada, decomposição, curvas de resposta, VIF, due-to | `mmm_dataset.csv` |
| 💰 Otimizador de Budget | Estratégico | Realocação ótima via SLSQP + cenário "E se" | modelo da página anterior |
| 🔀 Modelos de Atribuição | Tático | first/last-click, linear, time-decay, U-shaped; CPA por modelo | `digital_marketing_campaign_dataset.csv` |
| 🕸️ Markov e Shapley | Tático | Removal effect, Shapley Value, Sankey, comparação de todos os modelos | idem |
| 🎯 Propensão à Conversão | Tático | Classificação supervisionada, ROC/PR, SHAP, simulador de score | idem |
| 🧬 Calculadora A/B | Validação | Duração, impacto mid-range, tamanho de amostra, MDE | idem (pré-popula os inputs) |
| 📈 Simulador e Resultados | Validação | Z-test, Bayesiano, SPRT, veredito Winner/Neutral/Loser | digital + 200k campanhas |
| 🌍 Geo-Holdout × MMM | Validação | Holdout simulado, DiD, alerta de recalibração do MMM | `mmm_dataset.csv` + modelo |
| 🔁 Learning Repository | Governança | Registro de aprendizados, funil do framework, export CSV | session_state |

---

## Os três datasets

| Arquivo | Linhas | Granularidade | Uso |
|---|---|---|---|
| `mmm_dataset.csv` | 209 | semanal (2022–2025) | Fonte única do MMM. Mídia offline (TV, jornal) e digital (Instagram, Google Ads, YouTube, Influencer, OTT) + feriado, promoção, investimento da concorrência e `sales`. |
| `digital_marketing_campaign_dataset.csv` | 8.000 | cliente/campanha | Fonte do MTA (atribuição e propensão) e da pré-população da calculadora A/B. |
| `marketing_campaign_dataset.csv` | 200.000 | campanha (2021, diário) | Fonte do simulador A/B e validação cruzada do ranking de canais da MTA. |

---

## Limitações assumidas — leia antes de usar como referência

Este app é **pedagógico/demonstrativo**. Três aproximações são deliberadas e estão sinalizadas na
própria interface:

1. **Jornadas multi-touch são simuladas (Páginas 4, 5 e 10).**
   O dataset digital é por cliente/campanha: um canal por linha, sem sequência nem timestamp.
   Modelos de atribuição exigem uma sequência, então `src/mta/journey_sim.py` constrói uma jornada
   sintética **determinística** (sem aleatoriedade) a partir dos sinais de engajamento reais —
   `PreviousPurchases`, `LoyaltyPoints`, `SocialShares`, `EmailOpens/Clicks`, `WebsiteVisits`.
   As regras estão documentadas na docstring do módulo e num expander na Página 4.
   **Nada no alvo (`Conversion`) é inventado.** Com uma tabela `fct_mta_touchpoint` real, basta
   trocar `build_journeys()` — o resto do app continua igual.

2. **O geo-holdout é temporal, não geográfico (Página 8).**
   `mmm_dataset.csv` não tem recorte geográfico, então não há mercados de teste e controle.
   Simulamos um corte de investimento numa janela de semanas e usamos o próprio MMM como
   contrafactual. É a **mecânica** do teste de incrementalidade, não um experimento real — e o
   contrafactual herda todos os vieses do modelo que ele deveria auditar.

3. **A comparação A/B com dados reais é observacional (Página 7).**
   Comparar `Social Media` vs `PPC` no dataset não é um experimento randomizado: os grupos não
   foram sorteados. O p-value é calculado corretamente, mas responde "essas duas populações
   diferem?", não "essa mudança causou isso?".

Além disso: a **Base negativa** que pode aparecer na decomposição do MMM em forma Linear é
sinalizada na UI — é um sintoma real de má especificação quando todos os canais estão sempre
ativos, e a página sugere a correção (forma Log-Log, mais controles, alpha maior).

---

## Notas de implementação

- **Adstock geométrico:** `x_t^ad = x_t + decay · x_{t-1}^ad`.
- **Saturação Hill:** `y = x^s / (x^s + k^s)`, com preview da curva antes de aplicar.
- **Decomposição de contribuições:** aditiva no espaço do modelo (`coef_i · (z_i − z_i|spend=0)`),
  com a Base sendo a predição com toda a mídia zerada. A área empilhada soma exatamente o `sales`
  previsto nas três formas funcionais.
- **Otimizador:** o SLSQP roda em **frações do orçamento**, não em valores absolutos — em escala
  de 10⁷ os gradientes por diferenças finitas ficam na ordem de 10⁻⁹ e o otimizador para no ponto
  inicial achando que convergiu.
- **Removal Effect:** remover um canal significa redirecionar para `(null)` todas as transições que
  iam até ele. Apenas apagar o canal do caminho mantém a conversão e zera todos os removal effects.
- **Desbalanceamento (Página 10):** ~88% da base converte. Usamos `class_weight="balanced"`; a
  PR-AUC é a métrica de referência, não a accuracy. SMOTE seria a alternativa, não implementada
  para evitar dependência extra.
- O CSV original **nunca é sobrescrito** — todo tratamento é em memória / `st.session_state`.

---

## Sobre o documento de referência

O PPT `Framework_MMM_x_MTA_Digital_Analytics.pptx` não estava presente na pasta do projeto quando o
app foi construído. A linguagem, a ordem das camadas e a estrutura das calculadoras foram derivadas
da especificação em `MD_CODE_MMM_MTA_AB.md`. Se o PPT for adicionado em `reference/`, ele pode ser
lido na Home com `python-pptx` (já listado no `requirements.txt`).
