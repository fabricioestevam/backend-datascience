import pandas as pd
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Configuração do MongoDB
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client['odontolegal']
colecao = db['dados']

# Buscar dados do MongoDB
dados = list(colecao.find({}, {'_id': 0}))

if not dados:
    print("Nenhum dado encontrado no MongoDB!")
    exit(1)

print(f"Encontrados {len(dados)} casos no banco de dados.")

# Preparar dados para treinamento
lista = []
for d in dados:
    if 'vitima' in d and 'idade' in d['vitima'] and 'corEtnia' in d['vitima']:
        lista.append({
            "idade": d['vitima']['idade'],
            "corEtnia": d['vitima']['corEtnia'],
            "geolocalizacao": d['geolocalizacao'],
            "titulo": d['titulo'],
        })

df = pd.DataFrame(lista)
print(f"Dados preparados: {len(df)} registros válidos")
print("Colunas disponíveis:", df.columns.tolist())
print("Distribuição dos tipos de caso:")
print(df['titulo'].value_counts())

# Definir features e target
X = df[['idade', 'corEtnia', 'geolocalizacao']]
y = df['titulo']

# Encoder para as classes target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Definir features categóricas e numéricas
categorical_features = ['corEtnia', 'geolocalizacao']
numerical_features = ['idade']

# Criar preprocessador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Criar pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        use_label_encoder=False, 
        eval_metric='mlogloss'
    ))
])

# Treinar o modelo
print("Treinando o modelo...")
pipeline.fit(X, y_encoded)

# Salvar o modelo treinado
model_data = {
    'model': pipeline,
    'label_encoder': label_encoder
}

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model_data, model_file)

print("Modelo treinado e salvo em model.pkl!")

# Testar o modelo
print("\nTestando o modelo com um exemplo:")
exemplo = pd.DataFrame([{
    'idade': 25,
    'corEtnia': 'Parda',
    'geolocalizacao': 'Rua da Moeda'
}])

pred = pipeline.predict(exemplo)
pred_proba = pipeline.predict_proba(exemplo)
pred_label = label_encoder.inverse_transform(pred)[0]

print(f"Predição: {pred_label}")
print(f"Probabilidades: {dict(zip(label_encoder.classes_, pred_proba[0]))}")