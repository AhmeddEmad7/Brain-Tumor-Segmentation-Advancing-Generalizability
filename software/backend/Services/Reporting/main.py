import uvicorn
import json
import threading
import pika
import os
from app.schemas import RequestReport
from app.config import SessionLocal, engine
from app import crud
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
def wrap_report_text(report_text: str):
    """
    Wrap a plain text report into a Plate (Slate) compatible JSON structure.
    Here we create a single paragraph element with a unique id.
    """
    return [
        {
            "id": str(1),
            "type": "p",
            "children": [
                {
                    "text": report_text
                }
            ]
        }
    ]
def report_consumer_callback(ch,method,properties,body):
    try:
        print(f" [Report] Received {body.decode()}, starting processing...")
        body = json.loads(body.decode())
        study_uid = body['studyId']
        findings  = body['finding']
        wrapped_findings = wrap_report_text(findings)
        # Convert to JSON string before saving
        wrapped_findings_str = json.dumps(wrapped_findings)
        db = next(get_db())
        request_report = RequestReport(studyId=study_uid, content=wrapped_findings_str)
        # 1. save the report to the database
        created_report = crud.create_report(db, request_report)
        if not created_report:
            print(" [Report] Report not created")
        else:
            print(" [Report] Report created successfully")
        print(f" [Report] Report for study {study_uid} saved")
    except Exception as e:
        print(f" [Report] Error processing message: {e}")
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)     
           
def startconsumer():
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=os.getenv('RABBITMQ_HOST'),
                port=os.getenv('RABBITMQ_PORT'),
                credentials=pika.PlainCredentials(
                    os.getenv('RABBITMQ_USERNAME'),
                    os.getenv('RABBITMQ_PASSWORD')
                ))
        )
        channel = connection.channel()
        channel.queue_declare(queue='reporting_queue', durable=True)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue='reporting_queue', on_message_callback=report_consumer_callback)
        print("Reporting service consumer started. Waiting for messages...")
        channel.start_consuming()
        
if __name__ == "__main__":
    # Start the RabbitMQ consumer in a separate thread
    # consumer_thread = threading.Thread(target=startconsumer)
    # consumer_thread.daemon = True  # Daemonize thread
    # consumer_thread.start()
    uvicorn.run(
        app="app.app:app",
        host="0.0.0.0",
        port=9000,
        reload=True
    )
