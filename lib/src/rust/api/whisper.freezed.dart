// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'whisper.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$TranscriptionEvent {

 Object get field0;



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is TranscriptionEvent&&const DeepCollectionEquality().equals(other.field0, field0));
}


@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(field0));

@override
String toString() {
  return 'TranscriptionEvent(field0: $field0)';
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

@optionalTypeArgs TResult maybeMap<TResult extends Object?>({TResult Function( TranscriptionEvent_Progress value)?  progress,TResult Function( TranscriptionEvent_Success value)?  success,TResult Function( TranscriptionEvent_Failure value)?  failure,required TResult orElse(),}){
final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that);case TranscriptionEvent_Success() when success != null:
return success(_that);case TranscriptionEvent_Failure() when failure != null:
return failure(_that);case _:
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

@optionalTypeArgs TResult map<TResult extends Object?>({required TResult Function( TranscriptionEvent_Progress value)  progress,required TResult Function( TranscriptionEvent_Success value)  success,required TResult Function( TranscriptionEvent_Failure value)  failure,}){
final _that = this;
switch (_that) {
case TranscriptionEvent_Progress():
return progress(_that);case TranscriptionEvent_Success():
return success(_that);case TranscriptionEvent_Failure():
return failure(_that);}
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

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>({TResult? Function( TranscriptionEvent_Progress value)?  progress,TResult? Function( TranscriptionEvent_Success value)?  success,TResult? Function( TranscriptionEvent_Failure value)?  failure,}){
final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that);case TranscriptionEvent_Success() when success != null:
return success(_that);case TranscriptionEvent_Failure() when failure != null:
return failure(_that);case _:
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

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>({TResult Function( int field0)?  progress,TResult Function( List<TranscriptionSegment> field0)?  success,TResult Function( String field0)?  failure,required TResult orElse(),}) {final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that.field0);case TranscriptionEvent_Success() when success != null:
return success(_that.field0);case TranscriptionEvent_Failure() when failure != null:
return failure(_that.field0);case _:
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

@optionalTypeArgs TResult when<TResult extends Object?>({required TResult Function( int field0)  progress,required TResult Function( List<TranscriptionSegment> field0)  success,required TResult Function( String field0)  failure,}) {final _that = this;
switch (_that) {
case TranscriptionEvent_Progress():
return progress(_that.field0);case TranscriptionEvent_Success():
return success(_that.field0);case TranscriptionEvent_Failure():
return failure(_that.field0);}
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

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>({TResult? Function( int field0)?  progress,TResult? Function( List<TranscriptionSegment> field0)?  success,TResult? Function( String field0)?  failure,}) {final _that = this;
switch (_that) {
case TranscriptionEvent_Progress() when progress != null:
return progress(_that.field0);case TranscriptionEvent_Success() when success != null:
return success(_that.field0);case TranscriptionEvent_Failure() when failure != null:
return failure(_that.field0);case _:
  return null;

}
}

}

/// @nodoc


class TranscriptionEvent_Progress extends TranscriptionEvent {
  const TranscriptionEvent_Progress(this.field0): super._();
  

@override final  int field0;

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


class TranscriptionEvent_Success extends TranscriptionEvent {
  const TranscriptionEvent_Success(final  List<TranscriptionSegment> field0): _field0 = field0,super._();
  

 final  List<TranscriptionSegment> _field0;
@override List<TranscriptionSegment> get field0 {
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
  

@override final  String field0;

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

// dart format on
