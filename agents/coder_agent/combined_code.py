# Auto-generated Python code



def move_snake(snake):
    if not snake:
        return []

    head = snake[0]
    dx, dy = 1, 0

    new_snake = [(head[0] + dx, head[1] + dy)]
    for i in range(1, len(snake)):
        x = snake[i-1][0] + dx
        y = snake[i-1][1] + dy
        new_snake.append((x, y))

    return new_snake


def move_snake(snake):
    if not snake:
        return []

    head_x, head_y = snake[0]
    dx, dy = 1, 1

    new_snake = [(head_x + dx, head_y + dy)]

    for x, y in snake[1:]:
        new_snake.append((x + dx, y + dy))

    return new_snake


def move_snake(snake):
    if not snake:
        return []

    head_x, head_y = snake[0]
    dx, dy = 1, 0  # Moving to the right

    new_snake = [(head_x + dx, head_y + dy)]
    for i in range(1, len(snake)):
        x = snake[i-1][0] + dx
        y = snake[i-1][1] + dy
        new_snake.append((x, y))

    return new_snake


def move_snake(snake):
    if not snake:
        return []

    # Extract head position and movement direction
    head_x, head_y = snake[0]
    dx, dy = 1, 0  # Moving to the right

    new_snake = [(head_x + dx, head_y + dy)]

    # Update each subsequent segment based on previous positions
    for i in range(1, len(snake)):
        x = snake[i-1][0] + dx
        y = snake[i-1][1] + dy
        new_snake.append((x, y))

    return new_snake

# Example usage:
snake = [(0, 0), (1, 0), (2, 0)]
print(move_snake(snake))  # Output: [(1, 0), (2, 0), (3, 0)]


def move_snake(snake):
    if not snake:
        return []

    head_x, head_y = snake[0]
    dx, dy = 1, 0  # Moving to the right

    new_snake = [(head_x + dx, head_y + dy)]
    for i in range(1, len(snake)):
        x = snake[i-1][0] + dx
        y = snake[i-1][1] + dy
        new_snake.append((x, y))

    return new_snake


def move_snake(snake):
    if not snake:
        return []

    head = snake[0]
    direction = (1, 0)  # Move to the right for now

    dx, dy = direction
    new_snake = [(head[0] + dx, head[1] + dy)]
    for i in range(1, len(snake)):
        x = snake[i-1][0] + dx
        y = snake[i-1][1] + dy
        new_snake.append((x, y))

    return new_snake


def move_snake(snake):
    if not snake:
        return []

    # Extracting the current position of the snake's head (first element)
    x, y = snake[0]
    dx, dy = 1, 0  # Assuming movement to the right initially

    # Calculating the new position for each segment
    new_snake = [(x + dx, y + dy)]

    for i in range(1, len(snake)):
        next_x = snake[i][0] + dx
        next_y = snake[i][1] + dy
        new_snake.append((next_x, next_y))

    return new_snake


def move_snake(snake):
    if not snake:
        return []

    dx = 1  # Change in x-direction (e.g., right)
    dy = 0  # Change in y-direction (e.g., up or down)

    new_snake = [(snake[0][0] + dx, snake[0][1] + dy)]

    for x, y in snake[1:]:
        new_x = x + dx
        new_y = y + dy
        new_snake.append((new_x, new_y))

    return new_snake

# Example usage:
snake = [(0, 0), (1, 0), (2, 0)]
new_snake = move_snake(snake)
print(new_snake)  # Output: [(1, 0), (2, 0), (3, 0)]


snake = [(0, 0), (1, 0), (2, 0)]
dx, dy = 1, -1

# Extract head position
head_x, head_y = snake[0]

# Create new_snake with the moved head
new_snake = [ (head_x + dx, head_y + dy) ]

# Update each subsequent segment
for i in range(1, len(snake)):
    x = snake[i][0] + dx
    y = snake[i][1] + dy
    new_snake.append( (x, y) )

print(new_snake)
