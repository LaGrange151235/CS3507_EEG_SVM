import os

if __name__=="__main__":
    independent_path = "./independent/"
    dependent_path = "./dependent/"
    print("dependent")
    for record_name in os.listdir(dependent_path):
        record_path = dependent_path+record_name
        record_file = open(record_path)
        record_file_lines = []
        for line in record_file:
            record_file_lines.append(line)
        print(record_file_lines[0], record_file_lines[-1], record_file_lines[-2])

    print("\nindependent")
    for record_name in os.listdir(independent_path):
        record_path = independent_path+record_name
        record_file = open(record_path)
        record_file_lines = []
        for line in record_file:
            record_file_lines.append(line)
        print(record_file_lines[0], record_file_lines[-1], record_file_lines[-2])
