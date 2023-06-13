init-remote:
	ssh arena2 git clone git@github.com/ddorn/othello.git
	rsync -r data/ arena2:othello/data

bring-code-back:
	git ls-files | rsync -azP --files-from=- arena2:othello .
	ssh arena2 git status
	git status