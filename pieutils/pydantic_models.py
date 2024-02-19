from typing import Optional

from pydantic import BaseModel, Field

from typing import Dict


class AttributeSpec(BaseModel):
    """Relevant Meta Class to collect Attribute Information"""

    name: str = Field(..., description="Attribute Name")
    description: str = Field(...,
                             description="Specification of an attribute.")


class ProductCategorySpec(BaseModel):
    """Relevant Meta Class to collect Product Information"""

    name: str = Field(..., description="Product Category Name")
    description: str = Field(..., description="Explains why a product belongs to this category.")
    attributes: list[AttributeSpec] = Field(..., description="List of potential attributes of a product in this category")


class Attribute(BaseModel):
    """Relevant Meta Class to collect Attribute Information"""

    name: str = Field(..., description="Attribute Name")
    description: str = Field(...,
                             description="Specification of an attribute.")
    examples: Optional[list[str]] = Field(description="Example values of this attribute.")


class ProductCategory(BaseModel):
    """Relevant Meta Class to collect Product Information"""

    name: str = Field(..., description="Product Category Name")
    description: str = Field(..., description="Explains why a product belongs to this category.")
    attributes: list[Attribute] = Field(..., description="List of potential attributes of a product in this category")

class ProductSpecifications(BaseModel):
    """Relevant Meta class to collect product Specifications"""
    title: str = Field(..., description="Product Title")
    description: str = Field(..., description="Product Description")

class ErrorClass(BaseModel):
    """Relevant Meta class to array of Error Class"""
    error_type: str = Field(..., description="Error Class Name")
    general_guideline: str = Field(..., description="Instruction to avoid error")
    
    def to_dict(self):
        return {
            # Convert properties of ErrorClass to a dictionary format
            "error_type": self.error_type,
            "general_guideline": self.general_guideline
        }

class ErrorClasses(BaseModel):
    """Relevant Meta class to collect Error Classes"""
    error_classes: list[ErrorClass] = Field(..., description="List of potential error classes")

    def to_dict(self):
            # Convert the object to a dictionary
            return {
                "error_classes": [error_class.to_dict() for error_class in self.error_classes]
            }
    
class ErrorExample(BaseModel):
    title: Optional[str] = Field(description="Title of the product")
    description: Optional[str] = Field(description="Description of the product")
    category: str = Field(..., description="Product Category")
    target_scores: dict = Field(..., description="dict containing true attribute values")
    predictions: dict = Field(..., description="dict containing predicted attribute values")

class LLMResponse(BaseModel):
    attribute_errors: Dict[str, str] = {}
