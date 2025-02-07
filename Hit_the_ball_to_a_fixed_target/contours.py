import numpy as np

def find_contours_manual(edges):
    contours = []
    visited = np.zeros(edges.shape, dtype=bool)
    def dfs(x, y, current_contour):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if visited[cy, cx] or edges[cy, cx] == 0:
                continue
            visited[cy, cx] = True
            current_contour.append((cy, cx))
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < edges.shape[1] and 0 <= ny < edges.shape[0]:
                    stack.append((nx, ny))
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] == 255 and not visited[y, x]:
                current_contour = []
                dfs(x, y, current_contour)
                if len(current_contour) > 100:
                    contours.append(np.array(current_contour))
    return contours