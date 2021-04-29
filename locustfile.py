import time
from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    #wait_time = between(1, 2.5)

    @task
    def submitForm(self):
        self.client.post("/predict", {"season":"1","holiday":"0","workingday":"1","weather":"3","temp":"28.2","atemp":"34","humidity":"80","windspeed":"8","year":"2001","month":"9","day":"20","dayofweek":"7","hour":"20"})
