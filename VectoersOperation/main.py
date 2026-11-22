import math

def euclidean_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x**2 for x in vec))
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("vectors must have same length")

    dot_product = dot(vec1, vec2)

    mag1 = euclidean_norm(vec1)
    mag2 = euclidean_norm(vec2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)




def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:

   if len(vec1) != len(vec2):
       raise ValueError("vectors must have same length")

   result = []
   for i in range(len(vec1)):
       result.append(vec1[i] + vec2[i])
   return result


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("vectors must have same length")
    result = []
    for i in range(len(vec1)):
        result.append(vec1[i] - vec2[i])

    return result

def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("vectors must have same length")
    result = 0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]

    return result
