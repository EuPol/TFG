def read_float_values(file_path):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                value = float(line.strip())
                values.append(value)
            except ValueError:
                print("Skipping invalid value:", line.strip())
    return values

def calculate_statistics(values):
    if not values:
        return None, None, None, None, None
    
    max_value = max(values)
    min_value = min(values)
    mean_value = sum(values) / len(values)
    sorted_values = sorted(values)
    n = len(sorted_values)
    median_value = sorted_values[n // 2] if n % 2 != 0 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    value_range = max_value - min_value
    
    return max_value, min_value, mean_value, median_value, value_range

def main():
    file_path = "experiments/ncm_eu/distances.txt"
    values = read_float_values(file_path)
    max_val, min_val, mean_val, median_val, value_range = calculate_statistics(values)
    
    print("Maximum value:", max_val)
    print("Minimum value:", min_val)
    print("Mean value:", mean_val)
    print("Median value:", median_val)
    print("Value range:", value_range)

if __name__ == "__main__":
    main()
