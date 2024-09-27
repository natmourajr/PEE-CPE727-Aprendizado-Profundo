# Tabular Dockerfile
Git repository with Dockerfile to run Tabular database analysis

## Instructions to run

1 - First a docker image must be build
```bash
docker build . --tag=<login in dockerhub>/tabular:lastest
```

In this image, there is a directory called "workspace" (see [link](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda11.8-cudnn8-runtime/images/sha256-ff97981d417f43767865c977591c29e1ce35b076398d5c5122bdca4d2a454e1b?context=explore))

```bash
docker run --rm -it -v $(pwd):/workspace <login in dockerhub>/tabular:lastest 
```

Obs: In LPS, we have a different configuration. It is necessary to use singularity (see [link](https://github.com/natmourajr/docker))