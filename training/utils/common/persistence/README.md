# Persistence

The persistence package handles training persistence and resuming.

### Import package
```python
# Persistence Manager
from common.persistence import PersistenceManager

# Utility functions for getting distributed states.
from common.persistence import get_model_states, get_optimizer_states
```

### Before training, create a PersistenceManager and save training config. Call by all ranks!
```python
persistence = PersistenceManager("hdfs://path/to/save")
persistence.save_config(trainer_config)
```

### Check resume step
```python
resume = persistence.load_last_step()
if resume:
    print(resume.step)          # 500
    print(resume.models)        # {"model": PersistedModel}
    print(resume.optimizers)    # {"optimizer": PersistedOptimizer}
```

PersistedModel and PersistedOptimizer has the following APIs.
```python
resume.models["model"].config.path          # str
resume.models["model"].config.load()        # DictConfig
resume.models["model"].states.path          # str
resume.models["model"].states.load("cpu")   # state_dict
resume.models["model"].create("cpu")        # nn.Module
```


### Save model. Call by all ranks!
```python
persistence.save_model(
    step=1000,                              # Current step.
    name="model",                           # Model name. (Default to "model")
    config=trainer_config.unet.model,       # Model config.
    states=get_model_states(unet),          # Model states.
    dtype=None,                             # dtype conversion. Default None to disable.
    blocking=False,                         # Async copy. Default no blocking.
)
```
* You can save multiple models by giving them different model name (ema, etc.)
* The `get_model_states(model)` function helps unwrap DDP & FSDP & compile.


### Save optimizer. Call by all ranks!
```python
persistence.save_optimizer(
    step=1000,                              # Current step.
    name="optimizer",                       # Optimizer name. (Default to "optimizer")
    states=get_optimizer_states(optimizer), # Optimizer states.
    dtype=None,                             # dtype conversion. Default None to disable.
    blocking=False,                         # Async copy. Default no blocking.
)
```
* You can save multiple optimizers by giving them different optimizer name.
* The `get_optimizer_states(optimizer)` function helps unwrap ZeroRedundancyOptimizer.
* The `get_fsdp_optimizer_states(optimizer, model)` function helps unwrap FSDP optimizers.


### Load model utility

```python
from common.persistence import load_model_from_path, load_model_from_task

persisted_model = load_model_from_path("hdfs://path/to/save")
persisted_model = load_model_from_path("hdfs://path/to/save", name="model")
persisted_model = load_model_from_path("hdfs://path/to/save", name="model", step=1000)

persisted_model = load_model_from_task("d5ad7ed1c6acd81f")
persisted_model = load_model_from_task("d5ad7ed1c6acd81f", name="model")
persisted_model = load_model_from_task("d5ad7ed1c6acd81f", name="model", step=1000)

# You can access all the information.
persisted_model.config.path          # str
persisted_model.config.load()        # DictConfig
persisted_model.states.path          # str
persisted_model.states.load("cpu")   # state_dict

# Or instantiate the model.
model = persisted_model.create("cpu") # nn.Module
```
