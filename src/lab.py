
from subprocess import run
from main import main


proc = run(["conda", "create", "-n", "testenv", "python=3.6.9", "-y"])

run(["git", "clone","https://github.com/LuxiWang99/490_Deep_Learning.git"])


run(["cd", "490_Deep_Learning"])
run(["conda", "deactivate"])
run(["conda", "create", "-n", "testenv", "python=3.6.9", "-y"])
run(["conda", "activate", "dl-class"])
run(["gdown", "https://drive.google.com/uc\?id\=1DH1XMch9zCg6I-2_ReK7gWMB__x-fzg8"])

main()