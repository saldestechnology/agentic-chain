class Vector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

class Vector3:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z
    
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    
    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

class NotVector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        
    def __add__(self, other: 'NotVector') -> 'NotVector':
        return NotVector(self.x - other.x, self.y - other.y)
    
    def __repr__(self):
        return f"NotVector({self.x}, {self.y})"

d1 = Vector3(4, 3, 5)
d2 = Vector3(4, 3, 5)
d3 = d1 + d2
print(d3)

n1 = NotVector(2, 4)
v2 = Vector(2, 4)
n3 = n1 + v2
print(n3)

v1 = Vector(42, 1337)
v2 = Vector(13, 37)
v3 = v1 + v2
print(v3)