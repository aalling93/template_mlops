# Wandb
weights and biases, wandb, is used for logging, keeping track of the training, sweeping, and more... 



- Model registry: Place for storing and versioning our models during deployment. It uses the more general Artefact registry, i.e. a model is an artefact. therefore, 1) Create a model registry. 2) log the model as an artefact, 3) register/link it to the model registry we have created using the Link method [https://docs.wandb.ai/ref/python/artifact/#link]
    - Download model from registry: 
    
        ```python
        import wandb
        run = wandb.init()
        artifact = run.use_artifact('<entity>/model-registry/<my_registry_name>:<version>', type='model')
        artifact_dir = artifact.download("<artifact_dir>")
        model = MyModel()
        model.load_state_dict(torch.load("<artifact_dir>/model.ckpt"))
        ```
- Artefacts can contain metadata. 

