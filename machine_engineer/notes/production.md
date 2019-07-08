# Production
## 1. Intro to Deployment
### Machine learning Workflow
* Machine learning Workflow consists of
	* Explore& Process data
	* Modelling
	* Deployment
* Paths to Deployment
	* 1. record python into language of production env
		* This is not popular because it involve recoding into C++ or JAVA whcih takes time
	* 2. code preductive model markup language (PMML) or portable Format Analytics (PFA)
		* Somwhat popular, needing the developer to write according to the standards
	* 3. Model converted into a format that can be used in production automatically
		* these are the most popular. Populare models such as ONNX can be coverted automatically
* Products are deployed by user --> application(agent)-->endpoint(server)
		* rest API is a framwork for HTTP request and response
			* endpoint-->URL
			* HTTP method--> post/get/set
			* HTTP headers--> data format etc
			* Message --> main user data for inputting into a function
		* its the application's responsibility to format the request message and translate response msg
			* user data is usually JSON/CSV formatted
* container
	* collection of software to be used for a specific purpose
		* computations infastructure
		* OS
		* Engine eg docker
		* compositions
			* libraries 
			* application
	* Advantages of containers
		* Isolates app --> better security
		* Require only software to run the app --> less resources
		* maintainance easier
		* scalable
* Characteristics of Deployment and Modelling
	* Characteristics of Modelling
		* hyperparameter --> cant be estimated from data eg number of nodes
	* Characteristics of Deployment
		* model versioning --> saved as metadata, should allow one to indicate model version
		* Update and Routing --> update to increase performance
			* allow a test of model performace to be compared to other varients
			* Predictions
				* On demand --> online, real-time
				* Batch --> asynchronous, high volume, low latency
		* Monitoring --> monitor to measure performance matrix

	* Common machine learning platforms
		* Sagemaker
			* 15 built in algorithm
			* custom algorithm with Pytorch, TF, Apache MXNet, Spark, Chainer
			* Jupyter notebook
			* Auto model tuning
			* monitoring models
			* on demand type of predictions
		* Google ML Engine
		* Paperspace
			* GPU backed VM
		* Cloud Foundry







