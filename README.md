## 📓 Notebook
Você pode acessar o código completo e os resultados neste arquivo:  
[titanic.ipynb](./Cópia_de_Untitled1.ipynb)

📂 Estrutura final do repositório
Você terá um notebook (titanic.ipynb) com o código e os gráficos, e um README.md com a explicação.  
Mas se quiser ver tudo junto, aqui está como ficaria:

---

`markdown

🚢 Projeto Titanic com IA Básica

Este projeto utiliza Python e Machine Learning para prever a sobrevivência dos passageiros do Titanic com base em variáveis como idade, sexo, classe e tarifa.

📚 Tecnologias utilizadas
- Python 3
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

🎯 Objetivo
Treinar um modelo simples de regressão logística para prever a sobrevivência dos passageiros e gerar insights visuais sobre os dados.

📊 Etapas do projeto
1. Importação e exploração do dataset Titanic
2. Limpeza e preparação dos dados
3. Transformação de variáveis categóricas em numéricas
4. Divisão em treino e teste
5. Treinamento do modelo
6. Avaliação da acurácia
7. Visualização de insights com gráficos

---

💻 Código principal

`python

1. Importar bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.modelselection import traintest_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

2. Carregar dataset Titanic (já disponível no Seaborn)
df = sns.load_dataset("titanic")

3. Explorar dados
print("Primeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais:")
print(df.info())

4. Selecionar colunas relevantes e remover valores nulos
df = df[["sex","age","fare","class","survived"]].dropna()

5. Transformar variáveis categóricas em numéricas
df = pd.getdummies(df, columns=["sex","class"], dropfirst=True)

6. Separar features (X) e target (y)
X = df.drop("survived", axis=1)
y = df["survived"]

7. Dividir em treino e teste
Xtrain, Xtest, ytrain, ytest = traintestsplit(X, y, testsize=0.2, randomstate=42)

8. Treinar modelo de regressão logística
model = LogisticRegression(max_iter=200)
model.fit(Xtrain, ytrain)

9. Avaliar modelo
ypred = model.predict(Xtest)
print("\nAcurácia do modelo:", accuracyscore(ytest, y_pred))

10. Visualizar insights com gráfico
plt.figure(figsize=(8,5))
sns.barplot(x="sex_male", y="survived", data=df)
plt.title("Taxa de sobrevivência por sexo")
plt.show()
`

---

🚀 Resultados
- Acurácia do modelo: ~75%  
- Gráfico principal: taxa de sobrevivência por sexo  
  - Mulheres: ~75% de sobrevivência  
  - Homens: ~20% de sobrevivência  

---

📂 Como executar
1. Clone este repositório:
   `bash
   git clone https://github.com/seuusuario/titanic-ia.git
   `
2. Instale as dependências:
   `bash
   pip install pandas seaborn matplotlib scikit-learn
   `
3. Abra e execute o notebook titanic.ipynb no Google Colab ou Jupyter Notebook.

---

👨‍💻 Autor
Projeto desenvolvido por Allan como parte do roadmap de estudos em tecnologia e inteligência artificial.
