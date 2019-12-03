
from subprocess import call



# run(["git", "clone","https://github.com/LuxiWang99/490_Deep_Learning.git"])


# call(["conda", "create", "-n", "testenv", "python=3.6.9", "-y"])

# call(["cd", "490_Deep_Learning"])
# call(["conda", "deactivate"])

# call(["conda", "env", "update", "-n", "testenv", "-f", "environment.yml"])
# call(["conda","init","bash"])
# call(["source", "~/miniconda2/etc/profile.d/conda.sh"])
# call(["conda", "activate", "testenv"])
call(["gdown", "https://drive.google.com/uc\?id\=1DH1XMch9zCg6I-2_ReK7gWMB__x-fzg8"])
call(["unzip", "data.zip"])
from main import main

main()