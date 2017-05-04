import csv
import bcrypt
# from getpass import getpass
import sys

# username = 'rizzello'
# raw_password = '1234'
username = 'eduard'
raw_password = '5678'

# salt and hash password
salt = bcrypt.gensalt()
if sys.version_info[0] < 3:
    combo_password = raw_password + salt
    hashed_password = bcrypt.hashpw(combo_password, salt)
else:
    combo_password = raw_password + salt.decode('utf-8')
    hashed_password = bcrypt.hashpw(combo_password.encode('utf-8'), salt)

with open('users.csv', 'a') as csvfile:
    fieldnames = ['username', 'password', 'salt']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # writer.writeheader() #used only once (when first creating the CSV)
    writer.writerow(
        {
            'username': username,
            'password': hashed_password,
            'salt': salt
        }
    )


"""
#example of how to read all rows
with open('names.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['username'], row['password'])
"""
