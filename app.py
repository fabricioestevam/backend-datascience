from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pymongo import MongoClient
import random
from datetime import datetime, timedelta
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Configuração do MongoDB
MONGO_URI = "mongodb+srv://feestevamnascimentojr:0qb7tcgrGgi7wgy3@cluster0.lgnffp1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['odontolegal']
colecao = db['dados']

def gerar_dados_aleatorio(n=100):
    tipos_de_caso = ["Homicídio", "Acidente de Trânsito", "Desaparecimento", "Lesão Corporal", "Violência Doméstica", "Assalto", "Tráfico"]
    locais = ["Rua da Moeda", "Rua Bom Jesus", "Praça da Liberdade", "Avenida Paulista", "Parque Ibirapuera", "Estação Central", "Hospital Municipal"]
    etnias = ["Branca", "Preta", "Parda", "Amarela", "Indígena"]
    generos = ["Masculino", "Feminino"]
    casos = []
    base_date = datetime.now()
    
    for i in range(n):
        data_caso = base_date - timedelta(days=random.randint(0, 365))
        caso = {
            "data_do_caso": data_caso.date().isoformat(),
            "titulo": random.choice(tipos_de_caso),
            "geolocalizacao": random.choice(locais),
            "vitima": {
                "corEtnia": random.choice(etnias),
                "idade": random.randint(1, 90),
                "genero": random.choice(generos)
            }
        }
        casos.append(caso)
    return casos

@app.route('/api/casos', methods=['GET'])
def listar_casos():
    documentos = list(colecao.find({}, {'_id': 0}))
    return jsonify(documentos), 200

@app.route('/api/casos', methods=['POST'])
def criar_caso():
    data = request.get_json()
    if not data:
        abort(400, description="Dados inválidos")
    colecao.insert_one(data)
    return jsonify({"message": "Caso criado com sucesso!"}), 201

@app.route('/api/casos/<string:id_caso>', methods=['GET'])
def buscar_caso(id_caso):
    caso = colecao.find_one({"titulo": id_caso}, {'_id': 0})
    if not caso:
        abort(404, description="Caso não encontrado.")
    return jsonify(caso), 200

@app.route('/api/casos/<string:id_caso>', methods=['DELETE'])
def deletar_caso(id_caso):
    resultado = colecao.delete_one({"titulo": id_caso})
    if resultado.deleted_count == 0:
        abort(404, description="Caso não encontrado.")
    return jsonify({"message": "Caso deletado com sucesso!"}), 200

# Carregamento do modelo ML
model = None
label_encoder = None

def carregar_modelo():
    global model, label_encoder
    try:
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as model_file:
                data = pickle.load(model_file)
                model = data['model']
                label_encoder = data['label_encoder']
            print("Modelo carregado com sucesso!")
        else:
            print("Arquivo model.pkl não encontrado. Execute train_model.py primeiro.")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

@app.route('/api/predizer', methods=['POST'])
def predizer():
    if model is None:
        return jsonify({"erro": "Modelo não carregado"}), 500
        
    dados = request.get_json()
    if not dados or not all(k in dados for k in ("idade", "corEtnia", "geolocalizacao")):
        return jsonify({"erro": "JSON inválido. Esperado: idade, corEtnia, geolocalizacao"}), 400

    try:
        df = pd.DataFrame([dados])
        y_prob = model.predict_proba(df)[0]
        y_pred_encoded = model.predict(df)[0]
        y_pred = label_encoder.inverse_transform([y_pred_encoded])[0]
        classes = label_encoder.classes_

        resultado = {
            "classe_predita": y_pred,
            "probabilidades": {classe: round(prob, 4) for classe, prob in zip(classes, y_prob)}
        }
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"erro": f"Erro ao fazer predição: {str(e)}"}), 500

@app.route('/api/modelo/coeficientes', methods=['GET'])
def coeficientes_modelo():
    if model is None:
        return jsonify({"error": "Modelo não carregado"}), 500
        
    try:
        # Pegando o pré-processador e o classificador do pipeline
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']

        # Pegando nomes das features após o OneHotEncoding
        cat_encoder = preprocessor.named_transformers_['cat']
        categorical_features = ['corEtnia', 'geolocalizacao']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        numeric_features = ['idade']
        all_features = list(cat_features) + list(numeric_features)

        # Pegando as importâncias de feature do XGBoost
        importancias = classifier.feature_importances_

        features_importances = {
            feature: float(importance)
            for feature, importance in zip(all_features, importancias)
        }

        return jsonify(features_importances), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/estatisticas', methods=['GET'])
def estatisticas():
    documentos = list(colecao.find({}, {'_id': 0}))
    if not documentos:
        return jsonify({"message": "Sem dados na coleção"}), 400
    
    # Estatísticas básicas
    total_casos = len(documentos)
    
    # Distribuição por tipo
    tipos = {}
    idades = []
    etnias = {}
    locais = {}
    generos = {}
    
    for doc in documentos:
        # Contagem por tipo
        titulo = doc.get('titulo', 'Desconhecido')
        tipos[titulo] = tipos.get(titulo, 0) + 1
        
        # Idades para histograma
        if 'vitima' in doc and 'idade' in doc['vitima']:
            idades.append(doc['vitima']['idade'])
            
        # Etnias
        if 'vitima' in doc and 'corEtnia' in doc['vitima']:
            etnia = doc['vitima']['corEtnia']
            etnias[etnia] = etnias.get(etnia, 0) + 1
            
        # Locais
        local = doc.get('geolocalizacao', 'Desconhecido')
        locais[local] = locais.get(local, 0) + 1
        
        # Gêneros
        if 'vitima' in doc and 'genero' in doc['vitima']:
            genero = doc['vitima']['genero']
            generos[genero] = generos.get(genero, 0) + 1
    
    return jsonify({
        'total_casos': total_casos,
        'tipos': tipos,
        'idades': idades,
        'etnias': etnias,
        'locais': locais,
        'generos': generos
    }), 200

if __name__ == '__main__':
    # Carregar modelo ML
    carregar_modelo()
    
    # Inserir dados se necessário
    if colecao.count_documents({}) == 0:
        print("Inserindo dados aleatórios no MongoDB...")
        dados_iniciais = gerar_dados_aleatorio(100)
        colecao.insert_many(dados_iniciais)
        print(f"Inseridos {len(dados_iniciais)} casos no banco de dados.")
    
    app.run(debug=True)