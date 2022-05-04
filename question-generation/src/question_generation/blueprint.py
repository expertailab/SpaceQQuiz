from jsonschema import validate, ValidationError
from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest
from question_generation.question_generation import QuestionGeneration

blueprint = Blueprint('question_generation', __name__)

@blueprint.record_once
def record(setup_state):
    blueprint.question_generation = QuestionGeneration()
    print("QuestionGeneration loaded!")

@blueprint.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        json = request.get_json(force=True)
        validate(json, {
            'type': 'object',
            'required': ['text'],
            'properties': {
                'text': { 'type': 'string' }
            }
        })

        results = blueprint.question_generation.generate_questions(json['text'])
        return jsonify(results)

    except (BadRequest, ValidationError) as e:
        print('Bad request', e)
        return 'Bad request', 400

    except Exception as e:
        print('Server error', e)
        return 'Server error', 500
