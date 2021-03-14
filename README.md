# Machine Learning  

Four branches of machine learning In our previous examples, you’ve become familiar with three specific types of machine-learning problems: binary classification, multiclass classification, and scalar regression. All three are instances of supervised learning, where the goal is to learn the relationship between training inputs and training targets.
Supervised learning is just the tip of the iceberg—machine learning is a vast field with a complex subfield taxonomy. Machine-learning algorithms generally fall into four broad categories, described in the following sections.

## Supervised learning

**Supervised Learning** - *Is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. ... In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal).*

This is by far the most common case. It consists of learning to map input data to known targets (also called annotations), given a set of examples (often annotated by
humans). All four examples you’ve encountered in this book so far were canonical examples of supervised learning. Generally, almost all applications of deep learning 
that are in the spotlight these days belong in this category, such as optical character recognition, speech recognition, image classification, and language translation.
Although supervised learning mostly consists of classification and regression, there are more exotic variants as well, including the following (with examples): 

> - Sequence generation—Given a picture, predict a caption describing it. Sequence generation can sometimes be reformulated as a series of classification problems
(such as repeatedly predicting a word or token in a sequence). 
> - Syntax tree prediction—Given a sentence, predict its decomposition into a syntax tree.
> -  Object detection—Given a picture, draw a bounding box around certain objects inside the picture. This can also be expressed as a classification problem (given many candidate bounding boxes, classify the contents of each one) or as a joint classification and regression problem, where the bounding-box coordinates are predicted via vector regression.
> - Image segmentation—Given a picture, draw a pixel-level mask on a specific object.

Different Types of Supervised Learning Algorithms:

> 1. Decision Trees
> 1. Naive Bayes Classification
> 1. SVM - Support vector machines for classification problems
> 1. Random forest for classification and regression problems
> 1. Linear regression for regression problems
> 1. Ordinary Least Squares Regression
> 1. Logistic Regression
> 1. Ensemble Methods

## Unsupervised learning

- **Unsupervised Learning** - *Is a type of algorithm that learns patterns from untagged data. The hope is that through mimicry, the machine is forced to build a compact internal representation of its world*

This branch of machine learning consists of finding interesting transformations of the input data without the help of any targets, for the purposes of data visualization, data
compression, or data denoising, or to better understand the correlations present in the data at hand. Unsupervised learning is the bread and butter of data analytics, and
it’s often a necessary step in better understanding a dataset before attempting to solve a supervised-learning problem. Dimensionality reduction and clustering are well-known
categories of unsupervised learning.

Some popular examples of unsupervised learning algorithms are:
 
> 1. K-means for clustering problems
> 1. Apriori algorithm for association rule learning problems
> 1. PCA - Principal Component Analysis
> 1. SVC - Singular Value Decomposition
> 1. ICA - Independent Component Analysis

## Self-supervised learning

This is a specific instance of supervised learning, but it’s different enough that it deserves its own category. Self-supervised learning is supervised learning without
human-annotated labels—you can think of it as supervised learning without any humans in the loop. There are still labels involved (because the learning has to be
supervised by something), but they’re generated from the input data, typically using a heuristic algorithm.

For instance, autoencoders are a well-known instance of self-supervised learning, where the generated targets are the input, unmodified. In the same way, trying to predict
the next frame in a video, given past frames, or the next word in a text, given previous words, are instances of self-supervised learning (temporally supervised learning, in this case: supervision comes from future input data). Note that the distinction between supervised, self-supervised, and unsupervised learning can be blurry sometimes—these
categories are more of a continuum without solid borders. Self-supervised learning can be reinterpreted as either supervised or unsupervised learning, depending on whether
you pay attention to the learning mechanism or to the context of its application. 

NOTE In this book, we’ll focus specifically on supervised learning, because it’s by far the dominant form of deep learning today, with a wide range of industry applications. 

## Reinforcement learning

**Reinforcement Learning** - *Is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.*

Long overlooked, this branch of machine learning recently started to get a lot of attention after Google DeepMind successfully applied it to learning to play Atari
games (and, later, learning to play Go at the highest level). In reinforcement learning, an agent receives information about its environment and learns to choose actions that
will maximize some reward. For instance, a neural network that “looks” at a videogame screen and outputs game actions in order to maximize its score can be trained
via reinforcement learning. 

Currently, reinforcement learning is mostly a research area and hasn’t yet had significant practical successes beyond games. In time, however, we expect to see reinforcement
learning take over an increasingly large range of real-world applications: 

> - self-driving cars, 
> - robotics, 
> - resource management, 
> - education, 
> - and so on. 
 
![Machine Learning Segments](https://github.com/aridiosilva/AI/blob/main/machine-learning-topics-as.png)

## How to Select a Machine Learning Algorithm

A common question is “Which machine learning algorithm should I use?” The algorithm you select depends primarily on two different aspects of your data science scenario:

1 - What you want to do with your data? Specifically, what is the business question you want to answer by learning from your past data?
2 - What are the requirements of your data science scenario? Specifically, what is the accuracy, training time, linearity, number of parameters, and number of features your solution supports?

![how to](https://github.com/aridiosilva/AI/blob/8d87257f33d6e573da31ca4801f6022ef0789e73/HowToSelectMachineLearningAlgorithms.jpg)

### The Algorithm to choose Depeding On What You Want to do

![algorithms](https://github.com/aridiosilva/AI/blob/8d87257f33d6e573da31ca4801f6022ef0789e73/MicrosoftAzure-MachineLearningAlgorithmCheatSheet.jpg)

# Data Science 

Data analysis and machine learning have become an integrative part of the modern scientific methodology, offering automated procedures for the prediction of a phenomenon based on past observations, unraveling underlying patterns in data and providing insights about the problem. Yet, caution should avoid using machine learning as a black-box tool, but rather consider it as a methodology, with a rational thought process that is entirely dependent on the problem under study. In particular, the use of algorithms should ideally require a reasonable understanding of their mechanisms, properties and limitations, in order to better apprehend and interpret their results.
Topics and concepts related to AI, ML, DL, NN, DM, data analysis and data mining and others:

- **PCA - Principal Component Analysis** - One of the most often used **linear model** to extract information from data, and can be useful for you in term of data preprocessing, feature extraction, dimensional reduction, and data visualization.
- **Neural Networks**
- **Q Learning** - Pictorial introduction to Reinforcement Learning of Agent in unknown environment. Solving Tower of Hanoi.
- **K-Means Clustering** - Interesting algorithm to partition your data. Resources to classification analysis, clustering analysis, data mining
- **Random Forests** - Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees. 
- **Mixture Distribution** - Blend of independent component distributions.
- **Gaussian Mixture Model** - Flexible unsupervised learning for density estimation, clustering and likelihood optimization.
- **K-Nearest Neighbor Algorithm** - Supervised learning for classification, interpolation and extrapolation in data mining and statistical pattern recognition
- **Break Event Analysis** - Interactive way to optimize data
- **Stochastic Process** - Learn about Random Walk and Brownian Motion
- **Analysis of Algorithm** - Compare the efficiency of algorithms, rank the best, good and bad algorithms
- **AHP - Analytic Hierarchy Process** - Is a famous method for multi criteria decision making analysis
- **Market Basket Analysis**- Learn to discover association rules from transaction data of a store
- **MapReduce** - Frequently asked questions about MapReduce for Big Data
- **Non-Linear Transformation** - Transform nonlinear curves into simple linear regression
- **Recursive Average & Variance** - Efficient methods to compute average and variance using recursive formula for real time measurement data
- **Discriminant Analysis** - Classical algorithm for pattern recognition and classification analysis
- **Bootstrap Sampling** - Powerful Monte Carlo method to estimate distribution from sample's statistics
- **Digital Root** - Fascinating mathematical patterns of digital root
- **Feasibility Study** - Explain how to perform what-if scenario and sensitivity analysis
- **System Dynamic** - Introduction to system thinking terminologies such as rate, level, positive and negative feedback loop, causal and stock flow diagram
- **Similarity Measurement** - Basic knowledge on how to measure similarity and dissimilarity of performance index. Useful for clustering, Machine learning and data mining
- **Monte Carlo Simulation** - one of the largest and most important classes of numerical method for computer simulations
- **Mean and Average** - An eye opener to create many means or average beyond the traditional arithmetic, geometric and harmonic means. Fundamentalrelationship between averages.
- **Multiagent System** - Development of prototype multi agent simulation using only spreadsheet. Simple example of race of agents
- **Prime Factor** - Compute prime number and prime factors
- **Solving ODE** - Practical approach to solve Ordinary Differential Equation numerically using Euler and Runge-Kutta method
- **Kernel Regression** - Non-linear curve fitting technique using local kernel only by MS excel without macro
- **Linear Regression Model** - An introduction to regression model using MS Excel. Learn how to model, find the best-fit model and use graph, functions, matrix and Add Ins to automate the regression
- **Linear Algebra** - Understanding vector, matrix, solving linear equations, eigen value, and many more using online interactive programs
- **Data Analysis from Questionnaires** - Interactive descriptive statistics and contingency table and chi-square independent test to analyze your data from questionnaire survey
- **GIS** - Introduction to GIS with feature to use Arc GIS (Arc View, Arc Editor or Arc Info). Assumed you know nothing about GIS
- **Palindrome** - Learn how to test whether a word or a number is a palindrome. Generate and search palindrome.
- **Useful Summation & Tricks**
- **Growth Model** - Brief introduction to various basic growth model such as arithmetic , geometric, exponential and logistic growth phenomena using interactive program
- **Graph Theory** - A very gentle introduction using simple diagram of points and arcs
- **Difference Equation** - very fascinating subject of discrete dynamical system, its solution, behavior, equilibrium or critical value and stability
- **Ginger Bread Man** - About Chaos and having fun to produce Ginger Bread Man Cards
- **Fractal in Excel** - Produce fractal shape of Seipinski gasket
- **Adaptive Machine Learning Algorithm** - Histogram based learning formula with memory for educational Monte Calo game and simulation. Numerical example for hand calculation and MS Excel is explained
- **Continued Fraction** - Introduction to regular continued fraction (finite, infinite and periodic) and its application to convert decimal to fraction, compute Pi and Euler number
- **Queuing Theory** - Congestion and queuing problems
- **Generalized Inverse** - Solve regression using generalized inverse matrix
- **Page Rank** - How to compute Google Page Rank algorithm using MS Excel iteration and Matlab
- **Quadratic Function** - Explore the characteristics of quadratic function and parabola curve
- **Learning from Data** - Statistical concepts and online programs about central tendency and variation of data
- **Crypt Arithmetic** - Mathematical puzzle and solution
- **Complex Number** - Build Fractal Geometry using Complex Number - gives the basic arithmetic of Complex Number
- **EM Algorithm** - An iterative procedure to estimate the maximum likelihood of mixture density distribution
- **SVM - Support Vector Machine** - Classify non-linear data using SVM without programming
- **Maximum Likelihood** - Well-known method to estimate the parameter of a distribution
- **Simple Data Analysis** - Short Introduction of data analysis using Python
- **Numpy** - The basic data structure of array and matrices
- **Pandas** - The world most famous data analysis modules
- **Video Analysis using OpenCV-Python** - Super fun basic video analysis
- **SVM in Python** - Use scikit-learn for Support Vector Machine to do Machine Learning
- **Automatic Theorem Prover in Python**- Interactive way to use Python in solving first order proposition logic
- **Automatic Geocoding in Python** - Convert your tons of street, locations and cities into latitude and longitude coordinates
- **Practices Neural Network in Python** - Use Perceptron and Multi-Linear Preceptron to train and predict the data
- **Displaying Locations using Heatmap in Python** - Visualize the geocoded latitude and longitude coordinates into heatmap
-- **Ideal Flow using Python** - Network efficiency analysis based on Ideal Flow
-- **NLP - Natural Language Processing** - NLP using Python NLTK by analyzing the book of Psalm of David



# PCA - Principal Component Analysis 

In data science, one of the most often used **linear model** to extract information from data is **Principal Component Analysis (PCA)**. We need to learn about what PCA is and how PCA can be useful for you in term of:

   - data preprocessing, 
   - feature extraction, 
   - dimensional reduction, and 
   - data visualization.  
   
Note: Basic knowledge on Linear Algebra is necessary to understand the numerical examples and concepts involved. 

- What is PCA? 
- Why do we need PCA? 
- How does PCA works? 
- What are the PCA algorithms? 
- PCA Numerical Examples 
- How to compute Mean corrected data matrix? 
- How to calculate Standardized Data?
- How to calculate Covariance Matrix from your data? 
- How to calculate Correlation Matrix from your data? 
- How to obtain PCA from Covariance Matrix? 
- How to obtain PCA from Correlation Matrix? 
- How to obtain PCA based on Singular Value Decomposition (SVD)? 
- Free PCA online Interactive Programs 
- PCA in Microsoft Excel (using PCA Excel Add-In) 
- PCA in Python 
- How many PCA components should we use? 
- How to interpret PCA results? 
- What are the applications of PCA? 
- What are the strength and Weaknesses of PCA? 
- Brief Linear Algebra 
- Resources on PCA 

# Neural Networks

epending on the network architecture, the non-linear function inside the neuron and the learning methodsand its purposes, different name of neural network models was developed. 

-  What is Neural Network? 
   > - Why Use Neural Network?
   > - Limitation of Neural Network 
   > - Neural Network Terminologies and Notations 
- Model of a Neuron (NeuronModel.html)
   > - Aggregation Function 
   > - Activation Function 
   > - Bias and Dummy Input 
- Boolean Logic using Single Layer Neural Network 
- Input-Output Diagram of Perceptron
- Building Neural Network using Spreadsheet
- Training Single Layer Neural Network
- Advanced Training Single Layer Neural Network using Spreadsheet 
- Single Layer Bipolar Neural Network
- Multi-Layer Neural Network 
- Training Neural Network using Back Propagation
   > - Problems with Back Propagation 
- Training Neural Network Using Excel Solver
- Plug & Play Neural Network 
   > - Boolean Neural Network 
   > - Arithmetic Neural Network 
-  Neural Network for Regression Analysis
   > - Simple Linear Regression 
   > - Multiple Linear Regression   
   > - Logistic Regression
   > - Polynomial Regression 
- Applications of Neural Network
   > - Design of Neural Network 
   > - Training Neural Network   
   > - Neural Network Utilization 
- Classification Application: Prediction Beyond Expert System 
- Image Processing Application: Optical Number Recognition 
- Forecasting from Time Series Data 
- Summary: Integrated Approach to Neural Network 

