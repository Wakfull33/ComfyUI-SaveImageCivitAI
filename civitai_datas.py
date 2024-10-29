import hashlib
import argparse
import logging
import json
import os
import yaml

from safetensors import safe_open

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

CHECKPOINTS_PATH = ""
LORAS_PATH = ""
VAE_PATH = ""

DEBUG = False

BATCH_SIZE = 0
BATCH_IMAGE_REMAINING = 0

class LoraData:
    def __init__(self, name, weight, hash):
        self.name = name
        self.weight = weight
        self.hash = hash


class GenerationData:
    def __init__(self, datas, loraNode, loaderNode, samplerNode):
        self.preparePaths()

        self.loraStackerNodeID = None
        self.efficientLoaderNodeID = None
        self.samplerNodeID = None
        self.loraNodeName = loraNode
        self.loaderNodeName = loaderNode
        self.samplerNodeName = samplerNode

        self.getNodesIds(datas)
        self.loras = []

        self.getPrompts(datas)
        self.getSettings(datas)
        self.getModel(datas)
        self.getVAE(datas)
        self.getLoras(datas)

        self.tiHashes = []

        def __init__(self, datas, loraNode, loaderNode, samplerNode):
            self.preparePaths()

            self.loraStackerNodeID = None
            self.efficientLoaderNodeID = None
            self.samplerNodeID = None
            self.loraNodeName = loraNode
            self.loaderNodeName = loaderNode
            self.samplerNodeName = samplerNode

            self.getNodesIds(datas)
            self.loras = []

            self.getPrompts(datas)
            self.getSettings(datas)
            self.getModel(datas)
            self.getVAE(datas)
            self.getLoras(datas)

            self.tiHashes = []

    def getPrompts(self, datas):
        self.positivePrompt = datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('positive')
        self.negativePrompt = datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('negative')
        if DEBUG:
            print(f"Positive: {self.positivePrompt}\nNegative: {self.negativePrompt}")

    def getSettings(self, datas):
        global BATCH_SIZE
        global BATCH_IMAGE_REMAINING
        self.seed = datas.get(self.samplerNodeID, {}).get('inputs', {}).get('seed')
        self.steps = datas.get(self.samplerNodeID, {}).get('inputs', {}).get('steps')
        self.cfg = datas.get(self.samplerNodeID, {}).get('inputs', {}).get('cfg')
        self.sampler = datas.get(self.samplerNodeID, {}).get('inputs', {}).get('sampler_name')
        self.scheduleType = datas.get(self.samplerNodeID, {}).get('inputs', {}).get('scheduler')
        self.size = f"{datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('empty_latent_width')}x{datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('empty_latent_height')}"
        self.clipSkip = datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('clip_skip')

        if(BATCH_SIZE == 0):
            BATCH_SIZE = int(datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('batch_size'))
            BATCH_IMAGE_REMAINING = BATCH_SIZE

        if DEBUG:
            print(f"Seed: {self.seed}\tSteps: {self.steps}\tCFG: {self.cfg}\tSampler: {self.sampler}\tSchedule: {self.scheduleType}\tSize: {self.size}")

    def getModel(self, datas):
        self.modelName = datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('ckpt_name')
        self.modelHash = self.getModelHash(f'{CHECKPOINTS_PATH}/{self.modelName}')
        if DEBUG:
            print(f"modelName: {self.modelName}\nmodelHash: {self.modelHash}")

    def getVAE(self, datas):
        self.vaeName = datas.get(self.efficientLoaderNodeID, {}).get('inputs', {}).get('vae_name')
        self.vaeHash = self.getModelHash(f'{VAE_PATH}/{self.vaeName}')
        if DEBUG:
            print(f"vaeName: {self.vaeName}\nvaeHash: {self.vaeHash}")

    def getLoras(self, datas):

        count = int(datas.get(self.loraStackerNodeID, {}).get('inputs', {}).get('lora_count'))
        print(f"Lora Count: {count}")

        for i in range(1, count + 1):
            loraPathName = datas.get(self.loraStackerNodeID, {}).get('inputs', {}).get(f'lora_name_{i}')

            if loraPathName != "None":
                loraWeight = datas.get(self.loraStackerNodeID, {}).get('inputs', {}).get(f'lora_wt_{i}')

                safeTensorsFile = f'{LORAS_PATH}/{loraPathName}'
                # loraName = self.getLoraName(safeTensorsFile)
                loraName = loraPathName.split('/')[-1].split('.')[0]
                loraHash = self.getModelHash(safeTensorsFile)

                loraData = LoraData(loraName, loraWeight, loraHash)
                print(f"Lora n°{i}: Name: {loraData.name}\tWeight: {loraData.weight}\tHash: {loraData.hash}")

                self.loras.append(loraData)

    def getModelHash(self, path):
        tensors = {}
        if DEBUG:
            print(f"ModelPath: {path}")

        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()[:10]

    def getLoraName(self, path):
        print(f"Opening Lora file: {path}")
        with safe_open(path, framework="pt", device="cpu") as f:
            output = f.metadata()['ss_output_name']
            if output:
                return output
            else:
                logging.exception(f"Unable to get output from {path}")

    def formatLoras(self):
        formatted = ""

        for lora in self.loras:
            formatted += f"<lora:{lora.name}:{lora.weight}>\n"

        return formatted

    def preparePaths(self):
        global CHECKPOINTS_PATH
        global LORAS_PATH
        global VAE_PATH

        if os.path.exists(f"{folder_paths.base_path}/extra_model_paths.yaml"):
            with open(f"{folder_paths.base_path}/extra_model_paths.yaml", 'r') as file:
                config = yaml.safe_load(file)

                if config['a111']:
                    basePath = config['a111']['base_path']
                    CHECKPOINTS_PATH = f"{basePath}/{config['a111']['checkpoints']}"
                    LORAS_PATH = f"{basePath}/{config['a111']['loras']}".split('\n')[0]
                    VAE_PATH = f"{basePath}/{config['a111']['vae']}"

                elif config['comfyui']:
                    basePath = config['comfyui']['base_path']
                    CHECKPOINTS_PATH = folder_paths.folder_names_and_paths["checkpoints"]
                    LORAS_PATH = f"{basePath}/{config['comfyui']['loras']}"
                    VAE_PATH = f"{basePath}/{config['comfyui']['vae']}"

        else:
            CHECKPOINTS_PATH = f"{folder_paths.base_path}/models/checkpoints"
            LORAS_PATH = folder_paths.folder_names_and_paths["loras"]
            VAE_PATH = folder_paths.folder_names_and_paths["vae"]

        if DEBUG:
            print(f"CHECKPOINTS_PATH: {CHECKPOINTS_PATH}\tLORAS_PATH: {LORAS_PATH}\tVAE_PATH: {VAE_PATH}\t")

    def getNodesIds(self, datas):
        for node in datas:
            if self.loraNodeName in datas.get(node, {}).get('class_type', {}):
                self.loraStackerNodeID = node
            if self.loaderNodeName in datas.get(node, {}).get('class_type', {}):
                self.efficientLoaderNodeID = node
            if self.samplerNodeName in datas.get(node, {}).get('class_type', {}):
                self.samplerNodeID = node


    def __str__(self):
        has_loras = len(self.loras) > 0

        formatted = ""
        formatted += f"{self.positivePrompt}\n"

        if has_loras:
            formatted += f"{self.formatLoras()}\n"

        formatted += f"Negative prompt: {self.negativePrompt}\n"

        if self.steps and self.cfg and self.seed and self.size:
            formatted += f"Steps: {self.steps}, CFG scale: {self.cfg}, Seed: {self.seed}, Size: {self.size}, "

        if self.modelHash and self.modelName:
            formatted += f"Model hash: {self.modelHash}, Model: {self.modelName}, "

        if self.vaeHash and self.vaeName:
            formatted += f"VAE hash: {self.vaeHash}, VAE: {self.vaeName}, "

        if self.clipSkip:
            formatted += f"Clip slip: {self.clipSkip}, "

        if has_loras:
            formatted += "Lora hashes: \""

            for i in range(0, len(self.loras)):
                formatted += f"{self.loras[i].name}: {self.loras[i].hash}"
                if i < (len(self.loras) - 1):
                    formatted += ", "

            formatted += "\""

        return formatted

