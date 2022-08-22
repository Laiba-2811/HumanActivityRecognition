# HumanActivityRecognition
Abstract
Getting a good feature representation of data is paramount for Human Activity
Recognition (HAR) using wearable sensors. An increasing number of feature
learning approaches-in particular deep-learning based—have been proposed to
extract an effective feature representation by analyzing large amounts of data.We
implemented the codes and implementation details to make both the reproduction
of the results reported in this paper. Our work on UniMiB-SHAR dataset highlight
the effectiveness of deep-learning architectures involving Multi-Layer-Perceptron
(MLP), Convolutional Neural Network (CNN) to obtain features characterising in
the data.<br>
1 Introduction<br>
Human Activity Recognition (HAR) is a research topic which has attracted an increasing amount of
attention from the research community, in the wake of the development and spread of increasingly
powerful and affordable mobile devices or wearable sensors. The main goal of HAR is automatic
detection and recognition of activities from the analysis of data acquired by sensors. HAR finds
potential applications in numerous areas, ranging from surveillance tasks for security, to assistive
living and improvement of the quality of life or gaming.[1].<br>
1.1 UnimiB-SHAR:<br>
The UniMiB-SHAR dataset (University of Milano Bicocca Smartphone-based HAR) aggregates
data from 30 subjects (6 male and 24 female) acquired using the 3D accelerometer of a Samsung
Galaxy Nexus I9250 smartphone (S = 3). The data are sampled at a frequency of 50 Hz, and split in
17 different classes, comprising 8ADLs and 7 “falling” actions as shown in Table 1. Each activity is
either performed 2 or 6 times, with half of them having the subject place the smartphone in his/her
left pocket, and the other half in his/her right pocket.<br>
1.2 Multi-Layer Perceptron:<br>
Multi-Layer-Perceptron (MLP) is the simplest class of ANN, and involves a hierarchical organisa-
tion of neurons in layers. MLPs comprise at least three fully-connected layers (also called dense
layers) including an input, one or more intermediate (hidden) and an output layer, as shown in Fig-
ure 1. Each neuron of a fully-connected layer takes the outputs of all neurons of the previous layer
as its inputs. Considering that the output values of hidden neurons represent a set of features ex-
tracted from the input of the layer they belong to, stacking layers can be seen as extracting features
of an increasingly higher level of abstraction, with the neurons of the nth layer outputting features
computed using the ones from the (n 1)th layer.<br>
35th Conference on Neural Information Processing Systems (NeurIPS 2021).
1.3 Convolutional Neural Networks:<br>
CNNs comprise convolutional layers featuring convolutional neurons. The kth layer is composed of
nk neurons, each of which computes a convolutional map by sliding a convolutional kernel (f(k)T,
f(k)S) over the input of the layer (indicated in red in Figure 2). Convolutional layers are usually
used in combination with activation layers, as well as pooling layers. The neurons of the latter apply
a pooling function (e.g., maximum, average, etc.) operating on a patch of size (p(k)T, p(k)S) of
the input map to downsample it (indicated in blue in Figure 2, to make the features outputted by
neurons more robust to variations in temporal positions of the input data. Convolution and pooling
operations can either be performed on each sensor channel independently (f(k)S = 1 and/or p(k)S
= 1) or across all sensor channels (f(k)S = S and/or p(k)S = S).Similarly to regular dense layers,
convolutional-activation-pooling blocks can be stacked to craft high-level convolutional features.
Neurons of stacked convolutional layers operate a convolutional product across all the convolutional
maps of the previous layer. For classification purposes, a fully-connected layer after the last block
can be added to perform a fusion of the information extracted from all sensor channels. The class
probabilities are outputted by a softmax layer appended to the end of the network.
<br>
2 Related Work<br>
Traditionally, the fields of smart home technology and activity recognition are highly overlapping.
Smart homes provide the hardware to collect and process relevant data, while activity recognition
approaches extract semantic meanings from the collected data by adopting machine learning meth-
ods. Our system is not an exception to this overlap. For this reason, we want to point out highly
relevant work from the smart home area and research approaches from the field of activity recog-
nition. We read the following papers to have an extended level of understanding of background
knowledge and techniques used for the recognition of human activity recognition.<br>
3 Methodology<br>
In this project,we are using the feature encoding technique, any other researcher have not used the
feature encoding technique to extract the main features from the data set. They diectly applied CNN,
RNN, and Neural network.Before applying the super vector feature encoding technique,generate
codebooks for all sensor data in all dimensions. Codebooks are generated using local features,
which get through the sliding window technique, l is sliding window size and s stride.
3.1 Feature Extraction/ Local Descriptors:<br>
Low-level local features have become popular in action recognition due to their robustness to back-
ground clutter and independence in detection and tracking techniques. Local features extract through
the sliding window technique, the window size is 40, and the stride size is 10. Window slide on
11771 sample data through sliding we get the local feature of every row of every axis separately.<br>
3.2 Codebook Generation:<br>
Codebook Generation is implemented through GMM(Gaussian mixture models). A Gaussian mix-
ture model is a probabilistic model that assumes all the data points are generated from a mixture of
a finite number of Gaussian distributions with unknown parameters. A mean that defines its center.
3.3 Feature Encoding:<br>
The technique is using in this project is VLAD(Vector of Lo- cally Aggregated Descriptors), which
are Supervector-based encoding methods. The VLAD algorithm can be regarded as a simplified FV.
Its main method is to train a small codebook through a clustering method, subtracting the codebook
from the nearest visual codeword of input data X.<br>
3.4 Classification:<br>
Support Vector Machines (SVC) with a radial basis kernel; For classification, we are using the SVM
model to classify HAR. we are classifying Activities of Daily Living and Fall activities.<br>
3.5 MLP:<br>
We used a MLP with three hidden layers with Rectified Linear Units (RELU) activations, taking
vectors obtained by flattening frames of data as inputs. We noticed that the addition of a batch
normalization layer significantly improved the classification results. Batch normalization layers
standardize the inputs of the subsequent layers by computing parameters on batches of training data
to normalize each sensor channel independently.The final architecture used for our MLP models
thus consists of a batch normalization layer, followed by three fully-connected layers with RELU
activations, and an output layer with a softmax activation providing estimations of probabilities for
each class.<br>
3.6 CNN:<br>
Our CNN architecture involves three consecutive blocks, each including a convolutional, RELU ac-
tivation and max-pooling layers. Each convolutional kernel performs a 1D convolution over the time
dimension, for each sensor channel independently. Similarly to MLP, adding a batch normalization
layer right after the input layer yields significant performance improvements. The model selected
in the end comprises in order a batch normalization layer, three blocks of convolutional-RELU-
pooling layers, a fully-connected layer and a sigmoid and softmax layer for binary and Multiclass
respectively.<br>
