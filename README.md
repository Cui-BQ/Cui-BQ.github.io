## CSE 455 Final project report ##
I originally intended to improve the accuracy of a [picture classification from my CSE473 AI class project](https://courses.cs.washington.edu/courses/cse473/22wi/assignments/hw5/index.html). But I laterly thought I should try something new. Thus, I took some code from this picture classification project to do the Kaggle Bird classification project. 

In this project, I used Pytorch and Deep CNN to do the classification work. Since this dataset didn’t contain test data, I used 90% of the train data to train the CNN model and 10% to test the CNN model. 
So the procedure of my code is, first extra image data from the original dataset and make them to Pytorch Dataset with a random affine&HorizontalFlip on each image and resize them to 224x224, then use Pytorch DateLoader to load into batch size = 128. Then train the train data for 10 times while valid the test data after each train to see how the accuracy improved and if overfitting. I copied the train part from the CSE473 AI project and the rest of the bird classification project is my own work.

For this project, I first made a DeepCNN model with three Conv filters by myself, and trained on this model:

    class DeepCNN(nn.Module):
        def __init__(self, arr=[]):
            super(DeepCNN, self).__init__()
            
            self.conv1 = nn.Conv2d(3, 6, 3)           
            self.conv2 = nn.Conv2d(6, 12, 3)          
            self.conv3 = nn.Conv2d(12, 24, 3)         
            self.pool = nn.MaxPool2d(2)
            self.fc2 = nn.Linear(16224, 555)       

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))

            x = self.pool(x)
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            #x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
However, my CNN model was highly overfitted and the accuracy for both training and testing were low.
|      DeepCNN    | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 | epoch6 | epoch7 | epoch8 | epoch9 | epoch10 |
|  -------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------  |
|  train Accuracy | 0.0059 | 0.0212 | 0.0485 | 0.0852 | 0.1218 | 0.1614 | 0.1959 | 0.2349 | 0.2673 | 0.3017  |
|  valid Accuracy | 0.0101 | 0.0261 | 0.0441 | 0.0619 | 0.0770 | 0.0876 | 0.0845 | 0.0941 | 0.0936 | 0.0936  |

As you can see above, the best validation accuracy is 9.4%, while the best train accuracy is above 30%, which means the model is very bad.

Next I decided to use the pretrained “resnet18” model, it’s an image classification model from torchvision. 
    model = models.resnet18(pretrained=True)
    
The finaly result was good after epoch 10 times:
|      resnet18   | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 | epoch6 | epoch7 | epoch8 | epoch9 | epoch10 |
|  -------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------: |
|  train Accuracy | 0.1192 | 0.4138 | 0.5699 | 0.6588 | 0.7294 | 0.7822 | 0.8165 | 0.8467 | 0.8714 | 0.8871  |
|  valid Accuracy | 0.2885 | 0.4563 | 0.5206 | 0.5670 | 0.5924 | 0.6082 | 0.6193 | 0.6212 | 0.6432 | 0.6414  |

This new model is still overfitted, but the valid accuracy is much higher than my DeepCNN model. Then I used this model to predict the test data and submitted the prediction to the competition. I got a score of 0.64400, which is very close to my valid accuracy at epoch9 and 10.

The above two models were trained with resized 224x224 images, so I decided to quadruple the image size to 448x448 and use resnet18 to train again to see if what’s the improvement. And the outcome are:
|resnet18(488x488)| epoch1 | epoch2 | epoch3 | epoch4 | epoch5 | epoch6 | epoch7 | epoch8 | epoch9 | epoch10 |
|  -------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------: |
|  train Accuracy | 0.1224 | 0.4446 | 0.6161 | 0.7098 | 0.7690 | 0.8060 | 0.8363 | 0.8589 | 0.8776 | 0.8925  |
|  valid Accuracy | 0.3064 | 0.4959 | 0.5952 | 0.6401 | 0.6663 | 0.6852 | 0.7088 | 0.7080 | 0.7212 | 0.7243  |

The accuracy got higher with larger image size. But when I tried to predict the true 10000 test image, my code ran out of CUAD memory and failed. 

Here is my [Kaggle notebook](https://www.kaggle.com/code/cuiboqiang/notebook8868ae484c) for this project.
