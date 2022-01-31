# Feature Server
This repository provides the code to run a flask based API feature server.
It can be used to extract features for any specific layers of the network
or it may also be used to run inferencing on a predefined dataset.
While the feature server has been designed specifically for the **problem satement** explained
in the section below it can be adapted to different use cases with simple modifications.
This feature server can run any of the default pytorch models along with / without your
custom weights on multiple gpus in inference mode.
It extracts features from all the layers requested on the command line.
By default it runs on the imagenet dataset but can be used for any dataset as long as the image
information is provided in the needed CSV format.
It also uses the default imagenet training transforms to feed the images, you may replace it
with your own dataloader or transforms in the future.

NOTE: The feature server is meant to run in an infinite mode, that is, it will keep looping
on all images in the dataset and continue extracting features until you shutdown.
So please use responsibly keeping GPU greenhouse emissions to the minimum. :smiley:

Sample command
```
python main.py --logs_output_dir /tmp/logs --network_architecture resnet50 \
               --layer_names avgpool --batch_size 512 --workers 1 -vv
```

### _Problem Statement_
With advances in self-supervised learning it is a common practice to use representations from networks
trained on large datasets.
These representations are generally extracted for a users target dataset.
The representations can then be used as input to another network to make representations specific to
the target dataset in order to solve classification or another similar problem on the target dataset.
Generally, a MLP head is attached to the base network, the layers already trained with self supervised 
approach are set to `requires_grad=False` and then the entire network is trained.
Because the goal is to only train the MLP head, even though the forward pass is performed through the 
entire network the weight update during the back propagation is only performed for the MLP head.
This kind of an approach is very common in all self supervised papers, where after performing self supervised
training, in order to test the feature representation a MLP head is trained using this approach.
This pipeline is depicted in the figure below.

![Traditional Training Pipeline](Images/traditional_training.jpeg?raw=true "Traditional Training Pipeline")

Let us assume we wish to perform an abltation study with the MLP head based on either the hyper-parameters
like learning rate, architecture or the loss function.
The problem with the traditional approach is that we will have to perform these experiments sequentially and 
for each experiment we would not be able to re-use the features extracted from the representation network during the previous experiment.
One solution would be to pre-extract features in a file and load these features for each experiment.
The drawback of this approach is that when we use the traditional pipeline we may get `100` augmentations of each
sample if we are training for `100` epochs.
This means now we will have to save `100` features per sample in the file, which can be fairly expensive.

### _Solution_
The solution we propose to the above problem is to break the system into two parts the representation system and the MLP head.
The MLP head can fetch the data for each epoch or more samples asyncronously using the provided API's.
This architectural solution is detailed in the figure below.

![Feature Server Training Pipeline](Images/feature_server_pipeline.jpeg?raw=true "Feature Server Training Pipeline")



### _Design & Architecture_
The following diagram details the architecture followed by the code in this repository.

![Architecture details](Images/FeatureServer_Architecture.png?raw=true "Architecture details")


### _Available API's & Usage examples_

**Commandline**

Fetch the first entry in the Queue
```
watch -n 10 'curl http://localhost:9999/data'
```
Fetch all entries after timestamp
```
watch -n 10 "curl http://localhost:9999/after?timestamp=$(date -d '1 hour ago' +%s)"
```

**Python**

Fetch all entries after timestamp.
In this example we fetch all entries inserted in last one hour
```
import requests
from datetime import datetime, timedelta
last_hour_date_time = datetime.now() - timedelta(hours = 1)
data = requests.get(f"http://localhost:9999/after?timestamp={last_hour_date_time.timestamp()}")
```
