 /$$$$$$  /$$$$$$  /$$      /$$       /$$   /$$ /$$$$$$$$ /$$     /$$ /$$$$$$ 
|_  $$_/ /$$__  $$| $$$    /$$$      | $$  /$$/| $$_____/|  $$   /$$//$$__  $$
  | $$  | $$  \ $$| $$$$  /$$$$      | $$ /$$/ | $$       \  $$ /$$/| $$  \__/
  | $$  | $$$$$$$$| $$ $$/$$ $$      | $$$$$/  | $$$$$     \  $$$$/ |  $$$$$$ 
  | $$  | $$__  $$| $$  $$$| $$      | $$  $$  | $$__/      \  $$/   \____  $$
  | $$  | $$  | $$| $$\  $ | $$      | $$\  $$ | $$          | $$    /$$  \ $$
 /$$$$$$| $$  | $$| $$ \/  | $$      | $$ \  $$| $$$$$$$$    | $$   |  $$$$$$/
|______/|__/  |__/|__/     |__/      |__/  \__/|________/    |__/    \______/ 
                                                                              
Generating keys for IAM users to use the API

HELPFUL SITES:
http://www.trottercashion.com/2012/05/02/creating-aws-identity-and-access-management-signing-certificates.html
http://fortnightlabs.posterous.com/getting-started-with-ec2
http://www.robertsosinski.com/2008/01/26/starting-amazon-ec2-with-mac-os-x/

To generate keys for IAM users:
1) openssl genrsa -out pk-amazon-timir.pem 2048
2) openssl req -new -x509 -key pk-amazon-timir.pem -out cert-amazon-timir.pem
-days 3650
3) openssl pkcs8 -topk8 -in pk-amazon-timir.pem -nocrypt > pk-temp.pem
4) mv pk-amazon-timir.pem pk-amazon-timir.pem.orig
5) cp pk-temp.pem pk-amazon-timir.pem

6) Upload the keys to the IAM tab for the user by copying the cert(!!!) file
contents to the "Security Credentials" tab and choose signing certificates
button

 /$$$$$$$$  /$$$$$$   /$$$$$$        /$$   /$$ /$$$$$$$$ /$$     /$$ /$$$$$$ 
| $$_____/ /$$__  $$ /$$__  $$      | $$  /$$/| $$_____/|  $$   /$$//$$__  $$
| $$      | $$  \__/|__/  \ $$      | $$ /$$/ | $$       \  $$ /$$/| $$  \__/
| $$$$$   | $$        /$$$$$$/      | $$$$$/  | $$$$$     \  $$$$/ |  $$$$$$ 
| $$__/   | $$       /$$____/       | $$  $$  | $$__/      \  $$/   \____  $$
| $$      | $$    $$| $$            | $$\  $$ | $$          | $$    /$$  \ $$
| $$$$$$$$|  $$$$$$/| $$$$$$$$      | $$ \  $$| $$$$$$$$    | $$   |  $$$$$$/
|________/ \______/ |________/      |__/  \__/|________/    |__/    \______/ 
                                                                             
Note that keys for API calls are different from keys for SSH into EC2 
instances. The former can be setup per user in IAM and the latter must 
be specified in EC2 settings. 

HELPFUL SITES:
http://www.markbartel.ca/2012/04/creating-key-pairs-for-amazon-ec2.html

Strictly speaking you can use part 1) here to generate certs for API calls
using signing credentials in IAM - then export EC2_CERT=cert and
EC2_PRIVATE_KEY=key - this could potentially replace the steps in the above
instructions for IAM KEYS. 

1) Generate x509 signing certs
openssl req -x509 -newkey rsa:2048 -passout pass:a -keyout kx -out cert
export EC2_CERT=cert
openssl rsa -passin pass:a -in kx -out key
export EC2_PRIVATE_KEY=key

2) Generate key pairs
ssh-keygen -b 2048 -t rsa -f aws-key

3) Upload the PUB KEY (!!!!) file as a keypair in EC2 key pairs 

4) Check key fingerprints with ec2-fingerprint-key (N.B. this will not macth
ssh-keygen -lf <pubkey.in>



CREATING DIFFERENT USERS ON EC2 INSTANCE

Have user generate keypair - ssh-keygen -b 2048 -t rsa -f OUTPUTFILENAME
Send us OUTPUTFILENAME.pub
scp OUTPUTFILENAME.pub to instance
cat OUTPUTFILENAME.pub >> /home/ec2-user/.ssh/authorized_keys
Have user test with ssh -i OUTPUTFILENAME-key ec2-user@HOSTNAME

Admin should ssh to ec2 instance 
sudo useradd -g ec2-user USERNAME
sudo su USERNAME
do keygen (see above) if pubkey not already sent
chmod 700 /home/USERNAME/.ssh
cat OUTPUTFILENAME.pub >> /home/USERNAME/.ssh/authorized_keys
chmod 600 /home/USERNAME/.ssh/authorized_keys

