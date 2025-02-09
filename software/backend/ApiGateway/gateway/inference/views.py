from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import pika
import json
import environ
import redis
import os

env = environ.Env()
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_db = int(os.getenv('REDIS_DB', 0))

client_redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

def start_connection():
    # establish connection with rabbitmq server
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=env('RABBITMQ_HOST'),
        port=env('RABBITMQ_PORT'),
        credentials=pika.PlainCredentials(
            env('RABBITMQ_USERNAME'),
            env('RABBITMQ_PASSWORD'),
        )
    ))

    channel = connection.channel()
    channel.queue_declare(queue='inference', durable=True)

    return connection, channel


@api_view(['POST'])
def segmentation_inference(request):
    study_uid = request.data.get('studyInstanceUid')
    sequences = request.data.get('sequences')

    print("study_instance_uid", study_uid)
    print("sequence mapping", sequences)

    if not study_uid:
        return Response(
            status=status.HTTP_400_BAD_REQUEST,
            data={'message': 'studyInstanceUid is required'}
        )

    connection, channel = start_connection()

    message = {
        'studyInstanceUid': study_uid,
        'sequences': sequences
    }

    # send the study_uid to the inference queue
    channel.basic_publish(
        exchange='',
        routing_key='inf_segmentation',
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        )
    )
    print(f" [x] Sent {study_uid} to segmentation inference queue")

    connection.close()

    return Response(
        status=status.HTTP_200_OK,
        data={'message': 'Sent to inference for processing successfully, check the results later'}
    )


@api_view(['POST'])
def motion_inference(request):
    study_uid = request.data.get('studyInstanceUid')
    series_uid = request.data.get('seriesInstanceUid')

    if not study_uid or not series_uid:
        return Response(
            status=status.HTTP_400_BAD_REQUEST,
            data={'message': 'studyInstanceUid and seriesInstanceUid are required'}
        )
    if client_redis.get(f"inference/{series_uid}",):
      return Response(
            status=status.HTTP_200_OK,
            data={'message': 'Request already processed recently'}
        )
      
    connection, channel = start_connection()

    message = {
        'studyInstanceUid': study_uid,
        'seriesInstanceUid': series_uid
    }

    channel.basic_publish(
        exchange='',
        routing_key='inf_motion_correction',
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        )
    )

    print(f" [x] Sent {study_uid} to motion inference queue")
    
    client_redis.set(f"inference/{series_uid}",series_uid, ex=60)
    
    connection.close()

    return Response(status=status.HTTP_200_OK, data={'message': 'motion inference'})


@api_view(['POST'])
def synthesis_inference(request):
    study_uid = request.data.get('studyInstanceUid')
    sequences = request.data.get('sequences')

    if not study_uid or not sequences:
        return Response(
            status=status.HTTP_400_BAD_REQUEST,
            data={'message': 'studyInstanceUid and sequences are required'}
        )

    connection, channel = start_connection()

    message = {
        'studyInstanceUid': study_uid,
        'sequences': sequences
    }

    channel.basic_publish(
        exchange='',
        routing_key='inf_sequence_synthesis',
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        )
    )

    print(f" [x] Sent {study_uid} to synthesis inference queue")

    return Response(status=status.HTTP_200_OK, data={'message': 'synthesis inference'})
