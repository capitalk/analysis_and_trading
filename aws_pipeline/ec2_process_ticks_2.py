from boto.services.service import Service
import os

class generate_hdf(Service):
    ProcessingTime = 1200
    
    Command = """python /home/ec2-user/analysis_and_trading/feature_extraction/extractFeatures.py -d /home/ec2-user/features/ /home/ec2-user/%s"""

    def process_file(self, in_file_name, msg):
        out_file_name = os.path.join(self.working_dir, 'OUT.HDF')
        command = self.Command % (in_file_name, out_file_name)
        os.system(command)
        return [(out_file_name, 'x-binary/hdf')]

