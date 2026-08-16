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

## Tema claro e escuro

O app suporta os dois. A troca é pelo controle nativo do Streamlit —
**☰ (canto superior direito) → Settings → Appearance → Light / Dark / System** — e a barra lateral
mostra qual está ativo. `base = "dark"` no `config.toml` define apenas o padrão de quem abre pela
primeira vez.

As duas paletas ficam em `[theme.light]` e `[theme.dark]` no `.streamlit/config.toml`. O CSS custom
e os gráficos Plotly leem `st.context.theme.type` em tempo de render (`styling.tokens()`), então
acompanham a escolha sem recarregar a página. As **cores de marca** (navy, teal, azul, dourado) são
as mesmas nos dois temas, em tons médios legíveis sobre branco e sobre o fundo escuro; só as
superfícies e o texto mudam.

---

## Páginas guiadas (para público leigo)

Duas páginas percorrem a escada clássica da analítica, com o jargão sempre traduzido e uma
simulação interativa no meio do caminho:

| Passo | Pergunta | O que a página entrega |
|---|---|---|
| 1 · Descritivo | O que está acontecendo? | Os fatos: investimento, vendas/conversões, período |
| 2 · Diagnóstico | Por que aconteceu? | As causas: contribuição por mídia, crédito por canal, fatores externos |
| 3 · Preditivo | O que vai acontecer? | **Simulação:** verba de R$ 1k a R$ 1M, escolha de mídias e horizonte |
| 4 · Prescritivo | O que devo fazer? | A decisão: onde colocar, de onde tirar, e como provar que está certo |

**🧭 MMM na prática** trabalha no horizonte de calendário — bimestre, trimestre, semestre, ano —
porque é assim que se decide verba de mídia, incluindo TV e jornal. A simulação aplica o
investimento adicional sobre o padrão das últimas N semanas e roda a predição de novo, então
adstock e saturação entram na conta: dobrar a verba não dobra o retorno, e o gráfico "quanto mais
invisto, quanto mais vendo" mostra exatamente onde a curva cruza a linha do empate.

**🧭 MTA na prática** trabalha no horizonte de operação — dia, semana, mês — porque é o ritmo de
quem mexe em lance e criativo. Como o dataset de MTA não é série temporal, a simulação usa o
**CPA implícito** de cada canal (vindo do modelo de atribuição) com um desconto de eficiência
conforme a verba se afasta do patamar histórico.

Três decisões de método que estão expostas na própria tela:

- **A verba simulada é sempre ADICIONAL** ao que já roda. Nas duas páginas o resultado é comparado
  com a operação atual do mesmo período — é o que evita a leitura de que uma verba pequena vai
  mudar o patamar do negócio.
- **Teto de concentração no MMM** (padrão 50%, ajustável). Com verbas pequenas diante do histórico
  a saturação mal aparece, e a alocação puramente ótima manda 100% numa mídia só. O teto é gestão
  de risco, não matemática do modelo — e está dito assim na tela.
- **O MTA não estima saturação de verdade.** O desconto de eficiência é uma aproximação prudente;
  quando a pergunta é "quanto cabe neste canal", a resposta correta vem do MMM.

---

## Recomendação acionável

As telas de MMM e MTA não param no diagnóstico: cada uma abre com um painel
**"com base nestes dados, o melhor cenário é investir em X e atenção com Y"**, com o valor a
realocar e o impacto estimado. A lógica está em `src/insights.py`, e as regras são explícitas:

| Tela | Critério de "investir" | Critério de "atenção" | O que informa |
|---|---|---|---|
| 🧪 MMM Modelagem | maior **retorno marginal** no nível atual de investimento, lido da curva de resposta | menor retorno marginal (ou canal que já não responde) | quanto realocar e as vendas incrementais estimadas, medidas rodando o cenário no próprio modelo |
| 💰 Otimizador de Budget | canal que mais recebe budget na solução do SLSQP | canal que mais perde | quanto colocar e quanto economizar, canal a canal, em valores |
| 🔀 Modelos de Atribuição | canal mais **subvalorizado** pelo last-click | canal mais supervalorizado | quanto o budget mudaria se seguisse a jornada inteira em vez do último clique |
| 🕸️ Markov e Shapley | idem, usando o crédito **algorítmico** como referência | idem | a leitura mais confiável da camada de MTA |

Duas escolhas de método que importam:

- O ranking do MMM usa **retorno marginal, não ROI médio**. O ROI médio dilui a saturação e faz um
  canal já esgotado parecer boa aposta; o marginal responde "o *próximo* real rende quanto aqui?".
- A realocação sugerida na página de modelagem é **conservadora por construção** (20% do
  investimento do canal de atenção). Mover pouco, medir, e só então mover mais — a recomendação é
  uma hipótese a validar no geo-holdout, não uma ordem de compra.

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

## Documento de referência

`reference/Framework_MMM_x_MTA_Digital_Analytics.pptx` (35 slides) é a fonte da linguagem, da ordem
das camadas e da estrutura das calculadoras. O app o lê em tempo de execução com `python-pptx`
(`src/reference.py`): a Home traz um navegador com busca textual em todos os slides, os números do
sumário executivo, a **matriz de decisão** do slide 07 e as **7 etapas do Roadmap de Testes A/B**
do slide 20.2 — as mesmas etapas usadas no funil do Learning Repository.

Correspondência direta entre o documento e o app:

| Slide do framework | Onde vive no app |
|---|---|
| 03 · MMM, incl. mídia digital (Google/Meta Ads) | MMM Modelagem — canais digitais no modelo, com nota de que a otimização tática é do MTA |
| 06 · Modelos de atribuição do GA4 | MTA Atribuição — os 5 heurísticos, com a distorção do last-click explícita |
| 06 · Data-Driven Attribution (Shapley/regressão) | MTA Markov e Shapley — o "DDA caseiro" |
| 07 · Matriz de decisão | Home — tabela com a página correspondente a cada cenário |
| 20.2 · As 7 etapas do Roadmap | Learning Repository — campo `etapa` e funil do framework |
| 20.3 · Geo-experiments / holdout (DiD) | Geo-Holdout × MMM |
| 20.5 · Ferramentas e métodos por tipo de teste | Simulador A/B — Z-test, Beta-Binomial e SPRT |
| 20.6 · Loop de governança | Home (diagrama) e Learning Repository |

Se o `.pptx` for removido de `reference/`, o app continua funcionando: o bloco de navegação do
documento simplesmente não aparece.
