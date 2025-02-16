Chatbot de Recuperação de Artigos Biomédicos
Este repositório contém um chatbot focado em recuperar e recomendar artigos científicos na área biomédica. A aplicação utiliza o PubMedBERT para gerar embeddings semânticos tanto das consultas quanto dos documentos, permitindo uma busca mais contextual mesmo sem correspondência exata de palavras-chave. Além disso, o projeto inclui uma interface de chatbot desenvolvida em Gradio, tornando a interação mais fácil e acessível para usuários que desejam pesquisar tópicos relacionados a oncologia, inteligência artificial em saúde, saúde digital e outras áreas afins.

Sumário
1 Principais Funcionalidades
2 Instalação e Configuração
3 Estrutura do Projeto
4 Uso do Sistema
5 Métricas de Avaliação
6 Possíveis Expansões
7 Contribuições

1 Principais Funcionalidades
Coleta de Artigos: Scripts que baixam e unificam artigos do PubMed em arquivos JSON.
Geração de Embeddings: Uso do modelo PubMedBERT para criar vetores representativos dos textos, permitindo comparação semântica entre consulta e documentos.
Chatbot em Gradio: Interface amigável onde o usuário insere consultas em linguagem natural e recebe os artigos mais relevantes em poucos segundos.
Cálculo de Métricas: Implementações de Precision@k, MAP e MRR para avaliar a qualidade da recuperação dos artigos.

2 Instalação e Configuração
Clonar o Repositório
git clone https://github.com/SeuUsuario/SeuRepositorio.git
cd SeuRepositorio
Criar Ambiente Virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
Instalar Dependências
pip install -r requirements.txt
Se não existir um requirements.txt, instale manualmente:
pip install biopython torch transformers scikit-learn gradio
Configurar Credenciais para PubMed (opcional)
Caso use a API do PubMed em scripts de coleta, defina Entrez.email = "seu_email_válido@example.com".

4 Uso do Sistema
Coleta de Artigos
Execute o script responsável por baixar e unificar artigos do PubMed (ex.: Merge articles.py e depois Correct merge articles.py). Os dados serão salvos em JSON na pasta data/.
Criação dos Embeddings
Rode o script que carrega os artigos, gera embeddings via PubMedBERT e salva os vetores (.pt ou outro formato). 
Rodar o Chatbot
Com os embeddings já criados, execute o arquivo principal do chatbot: Chatbot.py
A aplicação abrirá em um link local (fornecido pelo Gradio) onde você poderá digitar consultas (como “Artificial Intelligence in health” ou “cancer treatment”) e receberá artigos relevantes.

5 Métricas de Avaliação
As métricas implementadas incluem:
Precision@k: Mede quantos dos k primeiros resultados são relevantes.
Mean Average Precision (MAP): Considera a posição de cada resultado relevante no ranque, fornecendo uma visão global do desempenho.
Mean Reciprocal Rank (MRR): Avalia quão cedo aparece o primeiro documento relevante em cada lista de resultados.
Esse pacote de métricas oferece um retrato abrangente de como o chatbot se comporta ao priorizar artigos importantes para cada consulta.

6 Possíveis Expansões
Fine-tuning de PubMedBERT: Ajuste do modelo em subáreas específicas (oncologia, saúde digital etc.) para ganhar precisão semântica.
Integração de Ontologias: Uso de bases como MeSH ou UMLS para melhorar a identificação de sinônimos e termos correlatos.
Feedback de Usuários: Adicionar um mecanismo para que pesquisadores avaliem a pertinência dos artigos recomendados, reforçando o ranqueamento.
Ingestão de Outras Fontes: Expandir para bases como Scopus e Web of Science, tornando o acervo de artigos ainda mais abrangente.

7 Contribuições
Sinta-se à vontade para enviar issues, pull requests ou sugestões. Toda colaboração que aprimore a precisão, a usabilidade ou a manutenção do projeto é bem-vinda. Sugestões de estrutura de dados, optimizações de GPU/CPU e melhorias na interface do Gradio também são importantes!



