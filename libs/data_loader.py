class DataLoader:
    #transform file into list of sentences
    def file_to_list(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            sentences = file.read().splitlines()
        return sentences[0:100]
    
    #create lists of input and output sentneces with len < 60
    def load_data(self, input_file, output_file):
        input_data = self.file_to_list(input_file)
        output_data = self.file_to_list(output_file)
        assert(len(input_data) == len(output_data))
        input_data_sieved, output_data_sieved = [], []
        for i in range(len(input_data)):
            if len(input_data[i]) < 60:
                input_data_sieved.append(input_data[i])
                output_data_sieved.append(output_data[i])
        return (input_data_sieved, output_data_sieved)