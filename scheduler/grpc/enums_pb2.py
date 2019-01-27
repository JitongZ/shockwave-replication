# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: enums.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='enums.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0b\x65nums.proto*=\n\nDeviceType\x12\x12\n\x0eUNKNOWN_DEVICE\x10\x00\x12\x07\n\x03K80\x10\x01\x12\x08\n\x04P100\x10\x02\x12\x08\n\x04V100\x10\x03*S\n\tJobStatus\x12\x12\n\x0eUNKNOWN_STATUS\x10\x00\x12\n\n\x06QUEUED\x10\x01\x12\x0b\n\x07RUNNING\x10\x02\x12\r\n\tSUCCEEDED\x10\x03\x12\n\n\x06\x46\x41ILED\x10\x04\x62\x06proto3')
)

_DEVICETYPE = _descriptor.EnumDescriptor(
  name='DeviceType',
  full_name='DeviceType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_DEVICE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='K80', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='P100', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='V100', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=15,
  serialized_end=76,
)
_sym_db.RegisterEnumDescriptor(_DEVICETYPE)

DeviceType = enum_type_wrapper.EnumTypeWrapper(_DEVICETYPE)
_JOBSTATUS = _descriptor.EnumDescriptor(
  name='JobStatus',
  full_name='JobStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_STATUS', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='QUEUED', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RUNNING', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUCCEEDED', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FAILED', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=78,
  serialized_end=161,
)
_sym_db.RegisterEnumDescriptor(_JOBSTATUS)

JobStatus = enum_type_wrapper.EnumTypeWrapper(_JOBSTATUS)
UNKNOWN_DEVICE = 0
K80 = 1
P100 = 2
V100 = 3
UNKNOWN_STATUS = 0
QUEUED = 1
RUNNING = 2
SUCCEEDED = 3
FAILED = 4


DESCRIPTOR.enum_types_by_name['DeviceType'] = _DEVICETYPE
DESCRIPTOR.enum_types_by_name['JobStatus'] = _JOBSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


# @@protoc_insertion_point(module_scope)
