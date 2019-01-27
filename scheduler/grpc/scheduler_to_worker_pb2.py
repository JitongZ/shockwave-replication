# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scheduler_to_worker.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import enums_pb2 as enums__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='scheduler_to_worker.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x19scheduler_to_worker.proto\x1a\x0b\x65nums.proto\"2\n\x0fStartJobRequest\x12\x0e\n\x06job_id\x18\x01 \x01(\x04\x12\x0f\n\x07\x63ommand\x18\x02 \x01(\t\">\n\x10StartJobResponse\x12\x0e\n\x06job_id\x18\x01 \x01(\x04\x12\x1a\n\x06status\x18\x02 \x01(\x0e\x32\n.JobStatus2F\n\x11SchedulerToWorker\x12\x31\n\x08StartJob\x12\x10.StartJobRequest\x1a\x11.StartJobResponse\"\x00\x62\x06proto3')
  ,
  dependencies=[enums__pb2.DESCRIPTOR,])




_STARTJOBREQUEST = _descriptor.Descriptor(
  name='StartJobRequest',
  full_name='StartJobRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='job_id', full_name='StartJobRequest.job_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='command', full_name='StartJobRequest.command', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=42,
  serialized_end=92,
)


_STARTJOBRESPONSE = _descriptor.Descriptor(
  name='StartJobResponse',
  full_name='StartJobResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='job_id', full_name='StartJobResponse.job_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status', full_name='StartJobResponse.status', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=94,
  serialized_end=156,
)

_STARTJOBRESPONSE.fields_by_name['status'].enum_type = enums__pb2._JOBSTATUS
DESCRIPTOR.message_types_by_name['StartJobRequest'] = _STARTJOBREQUEST
DESCRIPTOR.message_types_by_name['StartJobResponse'] = _STARTJOBRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StartJobRequest = _reflection.GeneratedProtocolMessageType('StartJobRequest', (_message.Message,), dict(
  DESCRIPTOR = _STARTJOBREQUEST,
  __module__ = 'scheduler_to_worker_pb2'
  # @@protoc_insertion_point(class_scope:StartJobRequest)
  ))
_sym_db.RegisterMessage(StartJobRequest)

StartJobResponse = _reflection.GeneratedProtocolMessageType('StartJobResponse', (_message.Message,), dict(
  DESCRIPTOR = _STARTJOBRESPONSE,
  __module__ = 'scheduler_to_worker_pb2'
  # @@protoc_insertion_point(class_scope:StartJobResponse)
  ))
_sym_db.RegisterMessage(StartJobResponse)



_SCHEDULERTOWORKER = _descriptor.ServiceDescriptor(
  name='SchedulerToWorker',
  full_name='SchedulerToWorker',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=158,
  serialized_end=228,
  methods=[
  _descriptor.MethodDescriptor(
    name='StartJob',
    full_name='SchedulerToWorker.StartJob',
    index=0,
    containing_service=None,
    input_type=_STARTJOBREQUEST,
    output_type=_STARTJOBRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_SCHEDULERTOWORKER)

DESCRIPTOR.services_by_name['SchedulerToWorker'] = _SCHEDULERTOWORKER

# @@protoc_insertion_point(module_scope)
