import logging
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy.exc import IntegrityError

from app import db
from app.models import ModelCall, Post, TranscriptSegment
from shared.config import (
    Config,
    GroqWhisperConfig,
    LocalWhisperConfig,
    RemoteWhisperConfig,
    TestWhisperConfig,
)

from .transcribe import (
    GroqWhisperTranscriber,
    LocalWhisperTranscriber,
    OpenAIWhisperTranscriber,
    TestWhisperTranscriber,
    Transcriber,
)


class TranscriptionManager:
    """Handles the transcription of podcast audio files."""

    def __init__(
        self,
        logger: logging.Logger,
        config: Config,
        model_call_query: Optional[Any] = None,
        segment_query: Optional[Any] = None,
        db_session: Optional[Any] = None,
        transcriber: Optional[Transcriber] = None,
    ):
        self.logger = logger
        self.config = config
        self.transcriber = transcriber or self._create_transcriber()
        self.model_call_query = model_call_query or ModelCall.query
        self.segment_query = segment_query or TranscriptSegment.query
        self.db_session = db_session or db.session

    def _create_transcriber(self) -> Transcriber:
        """Create the appropriate transcriber based on configuration."""
        assert self.config.whisper is not None, (
            "validate_whisper_config ensures that even if old style whisper "
            "config is given, it will be translated and config.whisper set."
        )

        if isinstance(self.config.whisper, TestWhisperConfig):
            return TestWhisperTranscriber(self.logger)
        if isinstance(self.config.whisper, RemoteWhisperConfig):
            return OpenAIWhisperTranscriber(self.logger, self.config.whisper)
        if isinstance(self.config.whisper, LocalWhisperConfig):
            return LocalWhisperTranscriber(self.logger, self.config.whisper.model)
        if isinstance(self.config.whisper, GroqWhisperConfig):
            return GroqWhisperTranscriber(self.logger, self.config.whisper)
        raise ValueError(f"unhandled whisper config {self.config.whisper}")

    def _check_existing_transcription(
        self, post: Post
    ) -> Optional[List[TranscriptSegment]]:
        """Checks for existing successful transcription and returns segments if valid."""
        # First check for successful transcription
        existing_whisper_call = (
            self.model_call_query.filter_by(
                post_id=post.id,
                model_name=self.transcriber.model_name,
                status="success",
            )
            .order_by(ModelCall.timestamp.desc())
            .first()
        )

        if existing_whisper_call:
            self.logger.info(
                f"Found existing successful Whisper ModelCall {existing_whisper_call.id} for post {post.id}."
            )
            db_segments: List[TranscriptSegment] = (
                self.segment_query.filter_by(post_id=post.id)
                .order_by(TranscriptSegment.sequence_num)
                .all()
            )
            if db_segments:
                if (
                    existing_whisper_call.last_segment_sequence_num
                    == len(db_segments) - 1
                ):
                    self.logger.info(
                        f"Returning {len(db_segments)} existing transcript segments from database for post {post.id}."
                    )
                    return db_segments
                self.logger.warning(
                    f"ModelCall {existing_whisper_call.id} for post {post.id} indicates {existing_whisper_call.last_segment_sequence_num + 1} segments, but found {len(db_segments)} in DB. Re-transcribing."
                )
            else:
                self.logger.warning(
                    f"Successful ModelCall {existing_whisper_call.id} found for post {post.id}, but no transcript segments in DB. Re-transcribing."
                )
        else:
            self.logger.info(
                f"No existing successful Whisper ModelCall found for post {post.id} with model {self.transcriber.model_name}."
            )
        return None

    def _handle_existing_model_call(self, post: Post) -> Optional[ModelCall]:
        """Check for existing pending/failed ModelCall records and handle them appropriately."""
        # Check for any existing ModelCall with the same constraint values
        existing_call = (
            self.model_call_query.filter_by(
                post_id=post.id,
                first_segment_sequence_num=0,
                last_segment_sequence_num=-1,
                model_name=self.transcriber.model_name,
            )
            .order_by(ModelCall.timestamp.desc())
            .first()
        )

        if existing_call:
            if existing_call.status == "pending":
                self.logger.info(
                    f"Found existing pending ModelCall {existing_call.id} for post {post.id}. Reusing it."
                )
                # Reset retry attempts and update timestamp for this new attempt
                existing_call.retry_attempts += 1
                existing_call.timestamp = datetime.utcnow()
                existing_call.error_message = None
                self.db_session.commit()
                return existing_call
            elif existing_call.status in ["failed", "failed_permanent"]:
                self.logger.info(
                    f"Found existing failed ModelCall {existing_call.id} for post {post.id}. Updating to retry."
                )
                # Reset the failed call to pending for retry
                existing_call.status = "pending"
                existing_call.retry_attempts += 1
                existing_call.timestamp = datetime.utcnow()
                existing_call.error_message = None
                self.db_session.commit()
                return existing_call
        
        return None

    def transcribe(self, post: Post) -> List[TranscriptSegment]:
        """
        Transcribes a podcast audio file, or retrieves existing transcription.

        Args:
            post: The Post object containing the podcast audio to transcribe

        Returns:
            A list of TranscriptSegment objects with the transcription results
        """
        self.logger.info(
            f"Starting transcription process for post {post.id} using {self.transcriber.model_name}"
        )

        existing_segments = self._check_existing_transcription(post)
        if existing_segments is not None:
            return existing_segments

        # Check for existing ModelCall records that might conflict with the unique constraint
        current_whisper_call = self._handle_existing_model_call(post)
        
        if current_whisper_call is None:
            # Create a new ModelCall record for this transcription attempt
            try:
                current_whisper_call = ModelCall(
                    post_id=post.id,
                    model_name=self.transcriber.model_name,
                    first_segment_sequence_num=0,  # Placeholder, will be updated
                    last_segment_sequence_num=-1,  # Placeholder, indicates no segments yet
                    prompt="Whisper transcription job",  # Standardized prompt for Whisper calls
                    status="pending",
                )
                self.db_session.add(current_whisper_call)
                self.db_session.commit()  # Commit to get an ID for current_whisper_call if needed, and to mark it as pending

                self.logger.info(
                    f"Created new Whisper ModelCall {current_whisper_call.id} for post {post.id}."
                )
            except IntegrityError as e:
                self.logger.warning(
                    f"IntegrityError when creating ModelCall for post {post.id}: {e}. "
                    f"Attempting to find and reuse existing record."
                )
                self.db_session.rollback()
                
                # Try one more time to find an existing record - this handles race conditions
                current_whisper_call = self._handle_existing_model_call(post)
                if current_whisper_call is None:
                    # If we still can't find one, this is an unexpected error
                    self.logger.error(
                        f"Failed to create or find existing ModelCall for post {post.id} after IntegrityError. "
                        f"This may indicate a database issue."
                    )
                    raise e

        try:
            self.logger.info(
                f"Calling transcriber {self.transcriber.model_name} for post {post.id}..."
            )
            pydantic_segments = self.transcriber.transcribe(post.unprocessed_audio_path)
            self.logger.info(
                f"Transcription by {self.transcriber.model_name} for post {post.id} resulted in {len(pydantic_segments)} segments."
            )

            db_transcript_segments = []
            if pydantic_segments:
                for i, segment_data in enumerate(pydantic_segments):
                    start_time = round(segment_data.start, 1)
                    end_time = round(segment_data.end, 1)

                    db_segment = TranscriptSegment(
                        post_id=post.id,
                        sequence_num=i,
                        start_time=start_time,
                        end_time=end_time,
                        text=segment_data.text,
                    )
                    db_transcript_segments.append(db_segment)

                self.db_session.add_all(db_transcript_segments)

                # Update ModelCall with success status and details
                current_whisper_call.first_segment_sequence_num = 0
                current_whisper_call.last_segment_sequence_num = (
                    len(db_transcript_segments) - 1
                )
                current_whisper_call.response = (
                    f"{len(db_transcript_segments)} segments transcribed."
                )
                current_whisper_call.status = "success"
            else:
                # No segments produced, still a form of success but note it.
                current_whisper_call.response = "No segments produced by transcriber."
                current_whisper_call.status = "success"
                # first/last_segment_sequence_num remain 0/-1 as initialized

            self.db_session.commit()
            self.logger.info(
                f"Successfully stored {len(db_transcript_segments)} transcript segments and updated ModelCall {current_whisper_call.id} for post {post.id}."
            )
            return db_transcript_segments

        except Exception as e:
            self.logger.error(
                f"Transcription failed for post {post.id} using {self.transcriber.model_name}. Error: {e}",
                exc_info=True,
            )
            self.db_session.rollback()  # Rollback any potential partial additions from the try block

            call_to_update = (
                self.model_call_query.get(current_whisper_call.id)
                if current_whisper_call.id
                else None
            )
            if call_to_update:
                call_to_update.status = "failed_permanent"
                call_to_update.error_message = str(e)
                self.db_session.add(call_to_update)
                self.db_session.commit()
                self.logger.info(
                    f"Updated ModelCall {call_to_update.id} to status 'failed_permanent' for post {post.id}."
                )
            else:
                # This case should be rare if the initial commit of pending status was successful.
                self.logger.error(
                    f"Could not find ModelCall to update failure status for post {post.id} after transcription error."
                )

            raise
