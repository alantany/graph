import csv

def extract_first_n_lines(input_file, output_file, n=100000):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile, delimiter=' ')
        
        # 跳过注释行
        for line in infile:
            if not line.startswith('#'):
                break
        
        # 写入前 n 行非注释数据
        count = 0
        for line in infile:
            if count >= n:
                break
            if not line.startswith('#'):
                values = line.strip().split()
                csv_writer.writerow(values)
                count += 1
    
    print(f"已提取 {count} 行数据到 {output_file}")

if __name__ == "__main__":
    input_file = "roadnet-ca.txt"
    output_file = "roadnet-ca-100k.txt"
    extract_first_n_lines(input_file, output_file)