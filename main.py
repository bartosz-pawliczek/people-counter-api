from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import numpy as np
import requests

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def count_people(image):
    boxes, weights = hog.detectMultiScale(image, winStride=(4, 4))
    return len(boxes)


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/group_photo.jpg')
        count = count_people(img)
        return {'count': count}


class PeopleCounterLink(Resource):
    def get(self):
        url = request.args.get('url')
        if url:
            response = requests.get(url)
            if response.status_code == 200:
                nparr = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                count = count_people(img)
                return {'count': count}
            else:
                return {'error': 'Image could not be retrieved'}, 400
        return {'error': 'No URL provided'}, 400


api.add_resource(PeopleCounter, '/')
api.add_resource(PeopleCounterLink, '/link')

if __name__ == '__main__':
    app.run(debug=True)
