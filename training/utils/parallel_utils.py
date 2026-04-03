# copy from EasyVolumetricVideo
# Author: Zhen Xu https://github.com/dendenxu

from typing import Callable, List, Dict
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
# from training.utils.console_utils import *
from threading import Thread
import torch
from torch import Tensor
from typing import List
import imageio
import os


import asyncio
from functools import wraps

def async_call_func(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        # Use run_in_executor to run the blocking function in a separate thread
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

def _save_image_impl(save_img, save_path):
    """Common implementation for saving images synchronously or asynchronously"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, save_img)






def cat_dict(dict_list, dim=0, reserved_keys=['select_indice', 'dens_volume_fine', 'z_vals_fine', 'z_vals_mid_fine', 'points_fine', 'dens_volume', 'prob_volume', 't_vals', 't2z_func', 'img_feat', 't_mean_std_min_max']):
    return {k: torch.cat([item[k] for item in dict_list], dim) for k in dict_list[0] if isinstance(dict_list[0][k], Tensor) and k not in reserved_keys}

def cat_list(list_list: List[List[Tensor]], dim: int = 0):
    return [torch.cat([item[i] for item in list_list], dim=dim)  for i in range(len(list_list[0]))]

def cat_tensor(tensor_list: List[Tensor], dim=0):
    return torch.cat(tensor_list, dim=dim)

slice_func = lambda chunk_index, chunk_dim, chunk_size: [slice(None)] * chunk_dim + [slice(chunk_index, chunk_index+chunk_size)]
def chunkify(func, cat_func, chunk_tensors: List[Tensor], chunk_dim: int, chunk_size: int, **kwargs):
    '''
    func: function to be chunkified
    cat: function to concatenate the results
    chunk_tensors: list of tensors to be chunkified
    chunk_dim: dimension to be chunkified
    chunk_size: size of each chunk
    '''
    total_chunk_size = chunk_tensors[0].shape[chunk_dim]
    assert all([total_chunk_size == chunk_tensors[i].shape[chunk_dim] for i in range(1, len(chunk_tensors))])
    return cat_func([func(*[chunk_tensor[slice_func(i, chunk_dim, chunk_size)] for chunk_tensor in chunk_tensors], **kwargs) for i in range(0, total_chunk_size, chunk_size)], chunk_dim)
    
    all_ret = {}
    for i in range(0, chunk_size, chunk_num):
        ret = func(*[chunk_tensor[:, i:i + chunk_num] for chunk_tensor in chunk_tensors], **kwargs)
        if isinstance(ret, dict):
            for k in ret:
                if ret[k] is None:
                    continue
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        elif isinstance(ret, list) or isinstance(ret, tuple):
            for k in range(len(ret)):
                if ret[k] is None:
                    continue
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        elif isinstance(ret, torch.Tensor):
            if 0 not in all_ret:
                all_ret[0] = []
            all_ret[0].append(ret)
    if isinstance(ret, dict):
        return {k: torch.cat(all_ret[k], dim=chunk_dim) for k in all_ret}
    elif isinstance(ret, list) or isinstance(ret, tuple):
        return [torch.cat(all_ret[k], dim=chunk_dim) for k in all_ret]
    elif isinstance(ret, torch.Tensor):
        return torch.cat(all_ret[0], dim=chunk_dim)

def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper

@async_call
def save_image_async(save_img, save_path):
    """Save image asynchronously"""
    _save_image_impl(save_img, save_path)

def save_image(save_img, save_path):
    """Save image synchronously"""
    _save_image_impl(save_img, save_path)

def parallel_execution(*args, action: Callable, num_processes=32, print_progress=False, sequential=False, async_return=False, desc=None, **kwargs):
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        # TODO: Support all types of iterable
        action_kwargs = {key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == length else kwargs[key]) for key in kwargs}
        return action_args, action_kwargs

    if not sequential:
        # Create ThreadPool
        pool = ThreadPool(processes=num_processes)

        # Spawn threads
        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        if not async_return:
            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())  # will sync the corresponding thread
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), desc=desc, disable=not print_progress):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results
    
