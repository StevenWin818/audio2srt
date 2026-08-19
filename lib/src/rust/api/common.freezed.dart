// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'common.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$TranscriptionEvent {





@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent);
}


@override
int get hashCode => runtimeType.hashCode;

@override
String toString() {
  return 'TranscriptionEvent()';
}


}

/// @nodoc
class $TranscriptionEventCopyWith<$Res>  {
$TranscriptionEventCopyWith(TranscriptionEvent _, $Res Function(TranscriptionEvent) __);
}


/// Adds pattern-matching-related methods to [TranscriptionEvent].
extension TranscriptionEventPatterns on TranscriptionEvent {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>({TResult Function( TranscriptionEvent_Progress value)?  progress,TResult Function( TranscriptionEvent_ProgressDetail value)?  progressDetail,TResult Function( TranscriptionEvent_Success value)?  success,TResult Function( TranscriptionEvent_Failure value)?  failure,TResult Function( TranscriptionEvent_Segment value)?  segment,required TResult orElse(),}){
final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that);case TranscriptionEvent_ProgressDetail() when progressDetail != null:
return progressDetail(_that);case TranscriptionEvent_Success() when success != null:
return success(_that);case TranscriptionEvent_Failure() when failure != null:
return failure(_that);case TranscriptionEvent_Segment() when segment != null:
return segment(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>({required TResult Function( TranscriptionEvent_Progress value)  progress,required TResult Function( TranscriptionEvent_ProgressDetail value)  progressDetail,required TResult Function( TranscriptionEvent_Success value)  success,required TResult Function( TranscriptionEvent_Failure value)  failure,required TResult Function( TranscriptionEvent_Segment value)  segment,}){
final _that = this;
switch (_that) {
case TranscriptionEvent_Progress():
return progress(_that);case TranscriptionEvent_ProgressDetail():
return progressDetail(_that);case TranscriptionEvent_Success():
return success(_that);case TranscriptionEvent_Failure():
return failure(_that);case TranscriptionEvent_Segment():
return segment(_that);}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>({TResult? Function( TranscriptionEvent_Progress value)?  progress,TResult? Function( TranscriptionEvent_ProgressDetail value)?  progressDetail,TResult? Function( TranscriptionEvent_Success value)?  success,TResult? Function( TranscriptionEvent_Failure value)?  failure,TResult? Function( TranscriptionEvent_Segment value)?  segment,}){
final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that);case TranscriptionEvent_ProgressDetail() when progressDetail != null:
return progressDetail(_that);case TranscriptionEvent_Success() when success != null:
return success(_that);case TranscriptionEvent_Failure() when failure != null:
return failure(_that);case TranscriptionEvent_Segment() when segment != null:
return segment(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>({TResult Function( int field0)?  progress,TResult Function( PlatformInt64 processedMs,  PlatformInt64 totalMs)?  progressDetail,TResult Function( List<TranscriptionSegment> field0)?  success,TResult Function( String field0)?  failure,TResult Function( TranscriptionSegment field0)?  segment,required TResult orElse(),}) {final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that.field0);case TranscriptionEvent_ProgressDetail() when progressDetail != null:
return progressDetail(_that.processedMs,_that.totalMs);case TranscriptionEvent_Success() when success != null:
return success(_that.field0);case TranscriptionEvent_Failure() when failure != null:
return failure(_that.field0);case TranscriptionEvent_Segment() when segment != null:
return segment(_that.field0);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>({required TResult Function( int field0)  progress,required TResult Function( PlatformInt64 processedMs,  PlatformInt64 totalMs)  progressDetail,required TResult Function( List<TranscriptionSegment> field0)  success,required TResult Function( String field0)  failure,required TResult Function( TranscriptionSegment field0)  segment,}) {final _that = this;
switch (_that) {
case TranscriptionEvent_Progress():
return progress(_that.field0);case TranscriptionEvent_ProgressDetail():
return progressDetail(_that.processedMs,_that.totalMs);case TranscriptionEvent_Success():
return success(_that.field0);case TranscriptionEvent_Failure():
return failure(_that.field0);case TranscriptionEvent_Segment():
return segment(_that.field0);}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>({TResult? Function( int field0)?  progress,TResult? Function( PlatformInt64 processedMs,  PlatformInt64 totalMs)?  progressDetail,TResult? Function( List<TranscriptionSegment> field0)?  success,TResult? Function( String field0)?  failure,TResult? Function( TranscriptionSegment field0)?  segment,}) {final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that.field0);case TranscriptionEvent_ProgressDetail() when progressDetail != null:
return progressDetail(_that.processedMs,_that.totalMs);case TranscriptionEvent_Success() when success != null:
return success(_that.field0);case TranscriptionEvent_Failure() when failure != null:
return failure(_that.field0);case TranscriptionEvent_Segment() when segment != null:
return segment(_that.field0);case _:
  return null;

}
}

}

/// @nodoc


class TranscriptionEvent_Progress extends TranscriptionEvent {
  const TranscriptionEvent_Progress(this.field0): super._();
  

 final  int field0;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$TranscriptionEvent_ProgressCopyWith<TranscriptionEvent_Progress> get copyWith => _$TranscriptionEvent_ProgressCopyWithImpl<TranscriptionEvent_Progress>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent_Progress&&(identical(other.field0, field0) || other.field0 == field0));
}


@override
int get hashCode => Object.hash(runtimeType,field0);

@override
String toString() {
  return 'TranscriptionEvent.progress(field0: $field0)';
}


}

/// @nodoc
abstract mixin class $TranscriptionEvent_ProgressCopyWith<$Res> implements $TranscriptionEventCopyWith<$Res> {
  factory $TranscriptionEvent_ProgressCopyWith(TranscriptionEvent_Progress value, $Res Function(TranscriptionEvent_Progress) _then) = _$TranscriptionEvent_ProgressCopyWithImpl;
@useResult
$Res call({
 int field0
});




}
/// @nodoc
class _$TranscriptionEvent_ProgressCopyWithImpl<$Res>
    implements $TranscriptionEvent_ProgressCopyWith<$Res> {
  _$TranscriptionEvent_ProgressCopyWithImpl(this._self, this._then);

  final TranscriptionEvent_Progress _self;
  final $Res Function(TranscriptionEvent_Progress) _then;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? field0 = null,}) {
  return _then(TranscriptionEvent_Progress(
null == field0 ? _self.field0 : field0 // ignore: cast_nullable_to_non_nullable
as int,
  ));
}


}

/// @nodoc


class TranscriptionEvent_ProgressDetail extends TranscriptionEvent {
  const TranscriptionEvent_ProgressDetail({required this.processedMs, required this.totalMs}): super._();
  

 final  PlatformInt64 processedMs;
 final  PlatformInt64 totalMs;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$TranscriptionEvent_ProgressDetailCopyWith<TranscriptionEvent_ProgressDetail> get copyWith => _$TranscriptionEvent_ProgressDetailCopyWithImpl<TranscriptionEvent_ProgressDetail>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent_ProgressDetail&&(identical(other.processedMs, processedMs) || other.processedMs == processedMs)&&(identical(other.totalMs, totalMs) || other.totalMs == totalMs));
}


@override
int get hashCode => Object.hash(runtimeType,processedMs,totalMs);

@override
String toString() {
  return 'TranscriptionEvent.progressDetail(processedMs: $processedMs, totalMs: $totalMs)';
}


}

/// @nodoc
abstract mixin class $TranscriptionEvent_ProgressDetailCopyWith<$Res> implements $TranscriptionEventCopyWith<$Res> {
  factory $TranscriptionEvent_ProgressDetailCopyWith(TranscriptionEvent_ProgressDetail value, $Res Function(TranscriptionEvent_ProgressDetail) _then) = _$TranscriptionEvent_ProgressDetailCopyWithImpl;
@useResult
$Res call({
 PlatformInt64 processedMs, PlatformInt64 totalMs
});




}
/// @nodoc
class _$TranscriptionEvent_ProgressDetailCopyWithImpl<$Res>
    implements $TranscriptionEvent_ProgressDetailCopyWith<$Res> {
  _$TranscriptionEvent_ProgressDetailCopyWithImpl(this._self, this._then);

  final TranscriptionEvent_ProgressDetail _self;
  final $Res Function(TranscriptionEvent_ProgressDetail) _then;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? processedMs = null,Object? totalMs = null,}) {
  return _then(TranscriptionEvent_ProgressDetail(
processedMs: null == processedMs ? _self.processedMs : processedMs // ignore: cast_nullable_to_non_nullable
as PlatformInt64,totalMs: null == totalMs ? _self.totalMs : totalMs // ignore: cast_nullable_to_non_nullable
as PlatformInt64,
  ));
}


}

/// @nodoc


class TranscriptionEvent_Success extends TranscriptionEvent {
  const TranscriptionEvent_Success(final  List<TranscriptionSegment> field0): _field0 = field0,super._();
  

 final  List<TranscriptionSegment> _field0;
 List<TranscriptionSegment> get field0 {
  if (_field0 is EqualUnmodifiableListView) return _field0;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_field0);
}


/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$TranscriptionEvent_SuccessCopyWith<TranscriptionEvent_Success> get copyWith => _$TranscriptionEvent_SuccessCopyWithImpl<TranscriptionEvent_Success>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent_Success&&const DeepCollectionEquality().equals(other._field0, _field0));
}


@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(_field0));

@override
String toString() {
  return 'TranscriptionEvent.success(field0: $field0)';
}


}

/// @nodoc
abstract mixin class $TranscriptionEvent_SuccessCopyWith<$Res> implements $TranscriptionEventCopyWith<$Res> {
  factory $TranscriptionEvent_SuccessCopyWith(TranscriptionEvent_Success value, $Res Function(TranscriptionEvent_Success) _then) = _$TranscriptionEvent_SuccessCopyWithImpl;
@useResult
$Res call({
 List<TranscriptionSegment> field0
});




}
/// @nodoc
class _$TranscriptionEvent_SuccessCopyWithImpl<$Res>
    implements $TranscriptionEvent_SuccessCopyWith<$Res> {
  _$TranscriptionEvent_SuccessCopyWithImpl(this._self, this._then);

  final TranscriptionEvent_Success _self;
  final $Res Function(TranscriptionEvent_Success) _then;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? field0 = null,}) {
  return _then(TranscriptionEvent_Success(
null == field0 ? _self._field0 : field0 // ignore: cast_nullable_to_non_nullable
as List<TranscriptionSegment>,
  ));
}


}

/// @nodoc


class TranscriptionEvent_Failure extends TranscriptionEvent {
  const TranscriptionEvent_Failure(this.field0): super._();
  

 final  String field0;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$TranscriptionEvent_FailureCopyWith<TranscriptionEvent_Failure> get copyWith => _$TranscriptionEvent_FailureCopyWithImpl<TranscriptionEvent_Failure>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent_Failure&&(identical(other.field0, field0) || other.field0 == field0));
}


@override
int get hashCode => Object.hash(runtimeType,field0);

@override
String toString() {
  return 'TranscriptionEvent.failure(field0: $field0)';
}


}

/// @nodoc
abstract mixin class $TranscriptionEvent_FailureCopyWith<$Res> implements $TranscriptionEventCopyWith<$Res> {
  factory $TranscriptionEvent_FailureCopyWith(TranscriptionEvent_Failure value, $Res Function(TranscriptionEvent_Failure) _then) = _$TranscriptionEvent_FailureCopyWithImpl;
@useResult
$Res call({
 String field0
});




}
/// @nodoc
class _$TranscriptionEvent_FailureCopyWithImpl<$Res>
    implements $TranscriptionEvent_FailureCopyWith<$Res> {
  _$TranscriptionEvent_FailureCopyWithImpl(this._self, this._then);

  final TranscriptionEvent_Failure _self;
  final $Res Function(TranscriptionEvent_Failure) _then;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? field0 = null,}) {
  return _then(TranscriptionEvent_Failure(
null == field0 ? _self.field0 : field0 // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

/// @nodoc


class TranscriptionEvent_Segment extends TranscriptionEvent {
  const TranscriptionEvent_Segment(this.field0): super._();
  

 final  TranscriptionSegment field0;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$TranscriptionEvent_SegmentCopyWith<TranscriptionEvent_Segment> get copyWith => _$TranscriptionEvent_SegmentCopyWithImpl<TranscriptionEvent_Segment>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent_Segment&&(identical(other.field0, field0) || other.field0 == field0));
}


@override
int get hashCode => Object.hash(runtimeType,field0);

@override
String toString() {
  return 'TranscriptionEvent.segment(field0: $field0)';
}


}

/// @nodoc
abstract mixin class $TranscriptionEvent_SegmentCopyWith<$Res> implements $TranscriptionEventCopyWith<$Res> {
  factory $TranscriptionEvent_SegmentCopyWith(TranscriptionEvent_Segment value, $Res Function(TranscriptionEvent_Segment) _then) = _$TranscriptionEvent_SegmentCopyWithImpl;
@useResult
$Res call({
 TranscriptionSegment field0
});




}
/// @nodoc
class _$TranscriptionEvent_SegmentCopyWithImpl<$Res>
    implements $TranscriptionEvent_SegmentCopyWith<$Res> {
  _$TranscriptionEvent_SegmentCopyWithImpl(this._self, this._then);

  final TranscriptionEvent_Segment _self;
  final $Res Function(TranscriptionEvent_Segment) _then;

/// Create a copy of TranscriptionEvent
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') $Res call({Object? field0 = null,}) {
  return _then(TranscriptionEvent_Segment(
null == field0 ? _self.field0 : field0 // ignore: cast_nullable_to_non_nullable
as TranscriptionSegment,
  ));
}


}

// dart format on
