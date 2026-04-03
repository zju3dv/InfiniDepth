import re

def parse_func(file_path: str) -> None:
    """Parse function."""
    # Regular expression patterns to detect the start, content, and end of the table
    # start_end_pattern = re.compile(r"┏+┓|└+┘")
    title_pattern = re.compile(r"┃*┃")
    content_pattern = re.compile(r"│*│")

    # Flag to determine when inside the table
    inside_table = False
    table_lines = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            if re.match(title_pattern, line):
                print(line.strip())
                print('┡' + (len(line)-3) * '─' + '┩')
            if re.match(content_pattern, line):
                print(line.strip())
            # Check for start or end of table
    #         if re.match(start_end_pattern, line):
    #             print('1')
    #             if not inside_table:
    #                 # Starting the table
    #                 inside_table = True
    #             else:
    #                 # Ending the table, append line and stop reading further
    #                 table_lines.append(line)
    #                 break
    #             table_lines.append(line)
    #         elif inside_table and re.match(content_pattern, line):
    #             # Append content lines if inside table
    #             table_lines.append(line)

    # # Print the extracted table
    # for line in table_lines:
    #     print(line, end='')
