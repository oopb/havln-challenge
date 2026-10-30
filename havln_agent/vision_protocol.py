"""Explicit image units; never infer or clip a model's mixed coordinate system."""

import copy

from .api_client import APIError


COORDINATE_PROMPT = (
    ' Return concise JSON only, without Markdown or reasoning. Declare coordinate_space as '
    'normalized_xyxy for [left,top,right,bottom] in [0,1], or image_1000_xyxy for INTEGER '
    '[left,top,right,bottom] in [0,1000]. Use the same declared unit for EVERY box in this '
    'response, including components, people, regions, entry, treads and landings. Never mix '
    'fractional normalized coordinates with thousand-scale coordinates. Use null when unknown. '
    'Unknown text may be null or empty; unknown image regions use null boxes. Keep required '
    'booleans and direction categories in the declared schema. Keep evidence brief; avoid '
    'repeating the task or listing invisible components.')


def coordinate_schema(base, unit=None):
    normalized = copy.deepcopy(base)
    normalized['properties']['coordinate_space'] = dict(const='normalized_xyxy')
    thousand = copy.deepcopy(base)
    thousand['properties']['coordinate_space'] = dict(const='image_1000_xyxy')
    thousand['required'] = list(thousand['required']) + ['coordinate_space']

    def convert(node):
        if isinstance(node, dict):
            if node.get('minItems') == node.get('maxItems') == 4 and node.get('items', {}).get('maximum') == 1:
                node['items'] = dict(type='integer', minimum=0, maximum=1000)
            for child in node.values():
                convert(child)
        elif isinstance(node, list):
            for child in node:
                convert(child)
    convert(thousand)

    def nullable_text(node):
        if isinstance(node, dict):
            if node.get('type') == 'string' and 'maxLength' in node:
                node['type'] = ['string', 'null']
            for child in node.values():
                nullable_text(child)
        elif isinstance(node, list):
            for child in node:
                nullable_text(child)
    nullable_text(normalized)
    nullable_text(thousand)
    return (thousand if unit == 'image_1000_xyxy' else normalized if unit == 'normalized_xyxy'
            else dict(oneOf=[normalized, thousand]))


def normalize_coordinates(value, text_schema=None):
    result = copy.deepcopy(value)
    scale = 1000 if result.pop('coordinate_space', 'normalized_xyxy') == 'image_1000_xyxy' else 1

    def box(values, path):
        if values is None:
            return None
        converted = [v / scale for v in values] if scale != 1 else values
        if not (converted[0] < converted[2] and converted[1] < converted[3]):
            error = APIError('invalid_box_order: ' + '.'.join(map(str, path)), category='structured_field')
            error.validation_field = path
            raise error
        return converted

    def walk(node, path=()):
        if isinstance(node, dict):
            for key, child in list(node.items()):
                if key == 'box' or key.endswith('_box'):
                    node[key] = box(child, list(path) + [key])
                elif key == 'tread_boxes':
                    node[key] = [box(v, list(path) + [key, i]) for i, v in enumerate(child)]
                else:
                    walk(child, path + (key,))
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, path + (i,))
    walk(result)
    def text(node, schema):
        if node is None and schema.get('type') == 'string':
            return ''
        if isinstance(node, dict):
            return {key: text(child, schema.get('properties', {}).get(key, {}))
                    for key, child in node.items()}
        if isinstance(node, list):
            return [text(child, schema.get('items', {})) for child in node]
        return node
    if text_schema is not None:
        result = text(result, text_schema)
    return result


def coordinate_unit(task):
    return task.get('vision_contract', {}).get('coordinate_space') if isinstance(task, dict) else None


def coordinate_prompt(task):
    unit = coordinate_unit(task)
    if unit == 'image_1000_xyxy':
        return (' Return concise JSON with coordinate_space="image_1000_xyxy". Every box is '
                '[left,top,right,bottom] in INTEGER image coordinates from 0 to 1000. '
                'Use this single unit for people, obstacles, regions, entry, treads and landings. '
                'Never use fractional normalized coordinates. Unknown boxes are null; unknown '
                'evidence text is null or empty. Required booleans/categories follow the schema.')
    return COORDINATE_PROMPT
