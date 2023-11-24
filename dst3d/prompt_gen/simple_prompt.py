import os

from ..utils.imnet import imnet_synset_to_class


def generate_simple_prompt(num, class_name, a_prompt, n_prompt, **kwargs):
    prompts = [
        f'An image of {class_name}, {a_prompt}'
        for _ in range(num)]
    neg_prompts = [
        n_prompt
        for _ in range(num)]
    return prompts, neg_prompts


def generate_simple_synset_prompt(num, class_name, a_prompt, n_prompt, **kwargs):
    prompts = [
        f'An image of {imnet_synset_to_class(class_name)}, {a_prompt}'
        for _ in range(num)]
    neg_prompts = [
        n_prompt
        for _ in range(num)]
    return prompts, neg_prompts
