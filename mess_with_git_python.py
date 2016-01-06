# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:29:45 2016

@author: deanpospisil
"""

import os
from git import Repo

# rorepo is a a Repo instance pointing to the git-python repository.
# For all you know, the first argument to Repo is a path to the repository
# you want to work with
repo = Repo( '/Users/deanpospisil/Desktop/net_code' )
assert not repo.bare

git = repo.git
#parse this for status

#repo.git.add( 'somefile' )


modified = '\n\n\tmodified:'
status = git.status()

if modified in status:
    print('current file has not been commited')


sha = repo.head.commit.hexsha

abspath = os.path.abspath(__file__)
git.add(abspath)
git.commit(m='git python test')
