from flask import Flask
from gevent.pywsgi import WSGIServer
from question_generation.blueprint import blueprint as QuestionGenerationBlueprint

app = Flask(__name__)

app.register_blueprint(QuestionGenerationBlueprint)

if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 8081), app)
    http_server.serve_forever()
