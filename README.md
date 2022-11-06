## Training and deployment on Sagemaker

### Reason for using ml.t3.medium notebook instance:
![Sagemaker Instance](screenshots/1_Notebook_instance.png)

* Cost saving using this instance:
  * ml.t3.medium costs $0.05 an hour this can be seen on the sagemaker [pricing page](https://aws.amazon.com/sagemaker/pricing/)
* This instance saves as will only do few workloads like download of images and upload to S3. Other bulk like:
  * Training and Model deployments will depend on the compute power of instances ml.m5.xlarge used for both hyperparameter tuning and Estimator training. and ml.m5.large instance used for model deployment.	

### Download data to an S3 bucket

![S3 bucket](screenshots/2_S3_bucket_created.png)


### Training and Deployment
![Training jobs](screenshots/4_Training_Jobs.png)

This job `dog-pytorch-2022-11-05-19-00-13-634` is a training job trained using.

```py

### After puting parameters
estimator.fit({"training": "s3://sagemaker-bucket-05-11-2022/"}, wait=False)
```

After training, we deploy the endpoint

![Endopoint](screenshots/5_Deployed_Endpoint.png)

### Multi-instance Training

```py
estimator = PyTorch(
    entry_point='hpo.py',
    base_job_name='dog-pytorch',
    role=role,
    instance_count=3,
    instance_type='ml.m5.xlarge',
    framework_version='1.4.0',
    py_version='py3',
    hyperparameters=hyperparameters,
    ## Debugger and Profiler parameters
    rules = rules,
    debugger_hook_config=hook_config,
    profiler_config=profiler_config,
)
```
```py
estimator.fit({"training": "s3://sagemaker-bucket-05-11-2022/"}, wait=False)
```

![Multi -instance Training](screenshots/6_Multi_instance_Training.png)

**Deployed**

**Deployed new endpoint that was trained on multiple intances**

![New Endpoint](screenshots/7_Two_Deployed_Endpoint_.png)

## EC2 Training

![Launch EC2 Instance](screenshots/13_AMI_DeepLearning.png)

![EC2 Instance running](screenshots/14_EC2_Instance_running.png)

![EBS based Instance](screenshots/14_1_EBS_Based_Instance.png)
This is an EBS based instance

![Set up](screenshots/15_SomeSetup_Process.png)

```batch
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip

mkdir TrainedModels

vim solution.py

:set paste

python solution.py

cd TrainedModels
```

![Model Saved](screenshots/16_Trained_Model_Path.png)

This show the model trained model as seen above `model.pth`

Writing on EC2 instance is cheaper but also fast, but not userfriendly when compared to Sagemaker Notebook or Studio.

For on
## Lambda function setup

```py
import base64
import logging
import json
import boto3
#import numpy
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print('Loading Lambda function')

runtime=boto3.Session().client('sagemaker-runtime')
endpoint_Name='pytorch-inference-2022-11-05-20-43-15-461'#'BradTestEndpoint'

def lambda_handler(event, context):

    #x=event['content']
    #aa=x.encode('ascii')
    #bs=base64.b64decode(aa)
    print('Context:::',context)
    print('EventType::',type(event))
    bs=event
    runtime=boto3.Session().client('sagemaker-runtime')
    
    response=runtime.invoke_endpoint(EndpointName=endpoint_Name,
                                    ContentType="application/json",
                                    Accept='application/json',
                                    #Body=bytearray(x)
                                    Body=json.dumps(bs))
    
    result=response['Body'].read().decode('utf-8')
    sss=json.loads(result)
    
    return {
        'statusCode': 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'type-result':str(type(result)),
        'COntent-Type-In':str(context),
        'body' : json.dumps(sss)
        #'updated_result':str(updated_result)

        }
```

In the above code, `runtime.invoke_endpoint(...)`

## Security and testing

The Lambda function can only invoke my endpoint if there is proper security policies attached to it.

Attach a security policy to lambda function that allows it access to sagemaker endpoints.By attaching `AmazonSageMakerFullAccess` it grants this Lambda function access to the endpoints and other sagemaker services.

![Lambda Function](screenshots/8_Lambda_Function.png)
![Lambda Function IAM Role](screenshots/9_LambdafunctionIAM_Role.png)

**Test Event**
{ 
    "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg" 
    
}

**Response**

Test Event Name
Image-class-test

Response
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "text/plain",
    "Access-Control-Allow-Origin": "*"
  },
  "type-result": "<class 'str'>",
  "COntent-Type-In": "LambdaContext([aws_request_id=6389cc62-45ae-41ff-87ee-5eb8442d43b8,log_group_name=/aws/lambda/lambdafunction,log_stream_name=2022/11/05/[$LATEST]f3938c0938764b45902528fc8dc3299c,function_name=lambdafunction,memory_limit_in_mb=128,function_version=$LATEST,invoked_function_arn=arn:aws:lambda:us-east-1:690349917069:function:lambdafunction,client_context=None,identity=CognitoIdentity([cognito_identity_id=None,cognito_identity_pool_id=None])])",
  "body": "[[-5.44008207321167, -4.952992916107178, -1.5317052602767944, -0.5046547651290894, -3.427722454071045, -3.5470330715179443, -2.1158523559570312, -1.5338290929794312, -7.150924205780029, 0.23522217571735382, -0.6767486929893494, -4.5792131423950195, -2.448819875717163, 1.5442739725112915, -4.79373025894165, -3.6146109104156494, -7.377564907073975, -2.806501865386963, -3.56970477104187, 0.6937685608863831, -4.370532989501953, -2.9022064208984375, -6.072688102722168, -4.791072368621826, -4.495667934417725, -4.914665699005127, -0.3236158490180969, -1.770508885383606, -5.172568321228027, -2.3332908153533936, -4.425060272216797, -2.528372049331665, -5.196396350860596, -1.9796003103256226, -8.67808723449707, -7.016977310180664, -4.221601486206055, -1.8513274192810059, -1.2895582914352417, -2.932392120361328, -2.3384647369384766, -3.72230863571167, -0.4528353810310364, -2.516489267349243, -0.7855870127677917, -7.66058874130249, -1.2515116930007935, 0.217243492603302, -3.0159168243408203, -1.0865718126296997, -0.9226453304290771, -6.505958557128906, -7.088664531707764, -2.6274538040161133, -5.69277811050415, -2.5954782962799072, -4.136444091796875, -6.413861274719238, -2.9214165210723877, -1.8026350736618042, -7.051815986633301, -6.990673542022705, -7.743916034698486, -6.82808780670166, -2.714198112487793, -6.730476379394531, 0.8651849031448364, -5.2978010177612305, -1.652145504951477, -0.9539076685905457, 0.27124127745628357, -3.6734120845794678, -5.905435562133789, -3.464625358581543, -5.734074115753174, -1.474758267402649, -6.283289432525635, -1.4405708312988281, -3.6478824615478516, -3.242371082305908, -1.1309401988983154, -4.781346321105957, -0.8546137809753418, -0.9007806777954102, -6.97314977645874, -5.638144493103027, -1.275240421295166, -6.873162269592285, -3.3538429737091064, -2.3007302284240723, -5.834065914154053, -3.1364731788635254, -2.340365409851074, -6.831973552703857, -3.429497241973877, -3.3244616985321045, -3.2669477462768555, -5.467846393585205, -5.630269527435303, -6.411980152130127, -7.864996433258057, -3.741847515106201, -2.0103819370269775, -5.227265357971191, -5.439854145050049, -5.761460304260254, -3.2689850330352783, -2.0540072917938232, -1.9230411052703857, -0.7347715497016907, -3.558453321456909, -1.4514901638031006, -6.587024211883545, -5.59173583984375, -5.957094669342041, -2.426581621170044, -7.154305934906006, -0.4399915337562561, -4.310878276824951, -0.7948324680328369, -2.5780420303344727, -4.45313835144043, -2.9790825843811035, -3.404550313949585, -6.894444942474365, -5.507934093475342, -3.2638163566589355, -2.16451096534729, -3.645242691040039, -5.3472371101379395, -5.278973579406738, -1.2079694271087646, -5.8689680099487305]]"
}
These are measurement of likelihood from each types of dog from the specified list of classes.

**Any other Vunerabilities that needs to be addressed.**

## Concurrency and auto-scaling

### Concurrency

Unreserved account concurrency 1000 thus amount of lambda instances that I can use if I want to.

Provision concurrency 3 reserved concurrency.
Provision concurrency means that I am creating instances that will alway be on and alway be there to reply to requests.

![Provision Concurrency Ready](screenshots/10_Provision_Concurrency_Configured.png)

![Testing Lambda After Concurrency setup](screenshots/11_Testing_Lambda_After_Concurrency.png)

### Auto-Scaling

![Endpoint runtime setting](screenshots/12_Endpoint_runtime_setting.png)

Minimum instance count=1 and Maximum instance count=3
Target-value: Our endpoint need to decide when to reploy to traffic. Here if we recieve 100 invokations that is simultaneous, that will be a signal that it needs to create a new instance.

So this a very responsive  endpoint

Scale in cool down. will only take 30 seconds of elavated traffic for the endpoint to deploy more intances, and Scale outcool down is 30 seconds of decreased  traffic for the endpoint to delete its instances.
## References

1. [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)