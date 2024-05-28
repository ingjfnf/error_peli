from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from peliculas_des import predict_movie_genre  # Importar la función de predicción

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='1.0',
    title='Predicción de Géneros de Películas',
    description='API para predecir géneros de películas basados en la trama.'
)

ns = api.namespace('prediccion_genero', description='Predicción de géneros de películas')

entrada_modelo = api.model('Entrada', {
    'Titulo': fields.String(required=True, description='Título de la película'),
    'Trama': fields.String(required=True, description='Trama de la película'),
    'Ano': fields.Integer(required=True, description='Año de la película'),
})

resultado_modelo = api.model('Resultado', {
    'Titulo': fields.String,
    'Ano': fields.Integer,
    'Generos': fields.List(fields.String),
})

@ns.route('/')
class MovieGenrePredictionApi(Resource):
    @api.expect(entrada_modelo)
    @api.marshal_with(resultado_modelo)
    def post(self):
        data = request.json
        titulo = data['Titulo']
        trama = data['Trama']
        ano = data['Ano']
        
        try:
            titulo, ano, generos = predict_movie_genre(ano, titulo, trama)
            return {'Titulo': titulo, 'Ano': ano, 'Generos': generos}, 200
        except Exception as e:
            api.abort(404, f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
