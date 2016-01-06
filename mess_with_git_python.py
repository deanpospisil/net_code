# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:29:45 2016

@author: deanpospisil
"""

def provenance_commit(cwd):
    
    from os import system
    from git import Repo
    from datetime import datetime
    
    repo = Repo( cwd)
    assert not repo.bare
    
    #making a message. of when the commit was made
    time_str = str(datetime.now())
    time_str = 'provenance commit ' + time_str
    
    commit_message = "git commit -a -m  %r." %time_str
    system(commit_message)
    sha = repo.head.commit.hexsha
    
    return sha




