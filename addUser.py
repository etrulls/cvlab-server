import csv
import bcrypt
# from getpass import getpass

username = 'rizzello'
raw_password = '1234'

# salt and hash password
salt = bcrypt.gensalt()
combo_password = raw_password + salt
hashed_password = bcrypt.hashpw(combo_password, salt)


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
