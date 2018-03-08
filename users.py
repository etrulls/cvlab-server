import csv
import sys
from passlib.hash import sha256_crypt
import os

users = [
    {'username': '', 'password': ''},
]

do_header = False
if not os.path.isfile('users.csv'):
    do_header = True

with open('users.csv', 'a') as csvfile:
    fieldnames = ['username', 'password']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Add header the first time the file is created
    if do_header:
        writer.writeheader()

    for u in users:
        # Generate new salt, hash password
        password_hash = sha256_crypt.hash(u['password'])

        writer.writerow(
            {
                'username': u['username'],
                'password': password_hash
            }
        )
